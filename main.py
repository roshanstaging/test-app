#!/usr/bin/env python3
"""
WebRTC Streaming Server
Receives video/audio data from inference server and streams to frontend clients via WebRTC
"""

import asyncio
import json
import logging
import time
import queue
import threading
from typing import Dict, Set
import numpy as np
import cv2
import fractions
import requests
import os
from datetime import datetime

# WebRTC imports
from aiortc import VideoStreamTrack, AudioStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, \
    RTCIceServer, RTCIceCandidate
from aiohttp import web, WSMsgType
import aiohttp_cors
from av import VideoFrame, AudioFrame

# WebSocket client for receiving data from inference server
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ElevenLabs configuration
ELEVENLABS_API_KEY = "sk_42ec4dc4e85da379d30e1979421d3587a81cb026344d1dcf"  # Replace with your actual API key
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_VOICE_ID = "Qggl4b0xRMiqOwhPtVWT"  # Default voice ID, you can change this
AUDIO_FOLDER = "audio"

# Create audio folder if it doesn't exist
os.makedirs(AUDIO_FOLDER, exist_ok=True)


class StreamingVideoTrack(VideoStreamTrack):
    """Video track that streams frames received from inference server"""

    def __init__(self, fps=25):
        super().__init__()
        self.frame_queue = asyncio.Queue(maxsize=430)
        self.fps = int(fps)  # Ensure fps is an integer
        self.frame_count = 0
        self._is_active = True

        # DEBUG: Print track creation
        print(f"=== WEBRTC TRACK DEBUG ===")
        print(f"Created video track with FPS: {self.fps}")
        print(f"==========================")

    async def recv(self):
        """
        Pulls a video frame from the queue to be sent to the client.

        The main fix is here: The previous `asyncio.sleep` logic was removed.
        That sleep was creating an artificial delay, causing the slow-motion effect.
        Now, this function simply waits for the next frame to become available in the queue
        and sends it immediately. The pacing is correctly handled by the inference server
        and the WebRTC engine's handling of presentation timestamps (pts).
        """
        if not self._is_active:
            raise Exception("Track is stopped")

        logger.debug(f"Recv frame {self.frame_count}")

        try:
            # Dequeue the tuple: (pts, frame_data)
            pts, frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)  # <<< UPDATE THIS LINE

            # Create the VideoFrame object that aiortc needs.
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")

            # USE THE INCOMING PTS, DO NOT USE self.frame_count
            # The pts is in seconds (float), convert it to the timebase.
            # A 90kHz clock is standard for RTP video.
            time_base = fractions.Fraction(1, 90000)
            frame.pts = int(pts / time_base)  # <<< CRITICAL CHANGE
            frame.time_base = time_base  # <<< CRITICAL CHANGE

            # self.frame_count is no longer needed for timestamping
            return frame

        except asyncio.TimeoutError:
            logger.warning("Video frame queue was empty for 1s, sending black frame.")
            frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
            # For fallback, we can't know the right pts, but we must provide one.
            # This part is less critical as it's an error condition.
            frame.pts = int(time.time() * 90000)
            frame.time_base = fractions.Fraction(1, 90000)
            return frame
        except Exception as e:
            logger.error(f"Error in StreamingVideoTrack recv: {e}")
            # Provide a fallback black frame to be safe.
            frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
            frame.pts = int(time.time() * 90000)
            frame.time_base = fractions.Fraction(1, 90000)
            return frame
        # try:
        #     # Wait for a frame from the queue, with a timeout to avoid blocking indefinitely.
        #     try:
        #         frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
        #     except asyncio.TimeoutError:
        #         # If the queue is empty for a second, it's likely the stream has issues.
        #         # We send a black frame to keep the connection alive and prevent freezing.
        #         logger.warning("Video frame queue was empty for 1s, sending black frame.")
        #         frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        #
        #     # Create the VideoFrame object that aiortc needs.
        #     frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
        #
        #     # Assign a presentation timestamp (pts) and time_base.
        #     # This is crucial for the browser to render frames at the correct time.
        #     # We use a simple frame count for pts, which is a common and effective method.
        #     # frame.pts = self.frame_count
        #     frame.pts = self.frame_count *  (90000 // self.fps)
        #     # frame.time_base = fractions.Fraction(1, self.fps)
        #     frame.time_base = fractions.Fraction(1, 90000)  # Standard WebRTC timebase
        #
        #     # if self.frame_count % 30 == 0:
        #     #     print(f"DEBUG: Frame {self.frame_count}, FPS setting: {self.fps}, PTS: {frame.pts}")
        #
        #
        #     self.frame_count += 1
        #     return frame
        #
        # except Exception as e:
        #     logger.error(f"Error in StreamingVideoTrack recv: {e}")
        #     # In case of other errors, provide a fallback black frame to be safe.
        #     frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        #     frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
        #     # frame.pts = self.frame_count
        #     # frame.time_base = fractions.Fraction(1, int(self.fps))
        #     frame.pts = self.frame_count * (90000 // self.fps)
        #     frame.time_base = fractions.Fraction(1, 90000)
        #     self.frame_count += 1
        #     return frame

    async def add_frame(self, pts, frame_data):
        if not self._is_active:
            return

        logger.debug(f"Video queue size: {self.frame_queue.qsize()}")

        try:
            if frame_data.shape[0] != 480 or frame_data.shape[1] != 640:
                frame_data = cv2.resize(frame_data, (640, 480))

            item_to_queue = (pts, frame_data)  # <<< CREATE TUPLE

            try:
                # self.frame_queue.put_nowait(frame_data)
                self.frame_queue.put_nowait(item_to_queue)  # <<< QUEUE THE TUPLE
            except asyncio.QueueFull:
                logger.warning("Video queue full, dropping oldest frame")
                try:
                    self.frame_queue.get_nowait()
                    # self.frame_queue.put_nowait(frame_data)
                    self.frame_queue.put_nowait(item_to_queue)  # <<< QUEUE THE TUPLE
                except asyncio.QueueEmpty:
                    pass
        except Exception as e:
            logger.error(f"Error adding frame to queue: {e}")

    def stop(self):
        self._is_active = False


class StreamingAudioTrack(AudioStreamTrack):
    """Audio track that streams audio received from inference server"""

    def __init__(self, sample_rate=48000):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.samples_per_frame = 960  # 20ms @ 48kHz
        self.frame_count = 0
        self._is_active = True

    async def recv(self):
        if not self._is_active:
            raise Exception("Track is stopped")

        try:
            # Dequeue the tuple: (pts, audio_data)
            pts, audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)  # <<< UPDATE THIS LINE

            # Pad or trim audio data if necessary
            if len(audio_data) != self.samples_per_frame:
                if len(audio_data) < self.samples_per_frame:
                    audio_data = np.pad(audio_data, (0, self.samples_per_frame - len(audio_data)), mode='constant')
                else:
                    audio_data = audio_data[:self.samples_per_frame]

            # Construct frame correctly
            frame = AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono"
            )

            # USE THE INCOMING PTS
            # The timebase for audio is its sample rate.
            time_base = fractions.Fraction(1, self.sample_rate)
            frame.sample_rate = self.sample_rate
            frame.pts = int(pts / time_base)  # <<< CRITICAL CHANGE
            frame.time_base = time_base  # <<< CRITICAL CHANGE

            return frame

        except asyncio.TimeoutError:
            # Return silence if queue is empty
            audio_data = np.zeros(self.samples_per_frame, dtype=np.int16)
            frame = AudioFrame.from_ndarray(audio_data.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = int(time.time() * self.sample_rate)
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame
        except Exception as e:
            logger.error(f"Audio recv error: {e}")
            # Fallback to silence
            audio_data = np.zeros(self.samples_per_frame, dtype=np.int16)
            frame = AudioFrame.from_ndarray(audio_data.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = int(time.time() * self.sample_rate)
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            return frame

        # try:
        #     try:
        #         audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
        #     except asyncio.TimeoutError:
        #         audio_data = np.zeros(self.samples_per_frame, dtype=np.int16)
        #
        #     if len(audio_data) != self.samples_per_frame:
        #         if len(audio_data) < self.samples_per_frame:
        #             audio_data = np.pad(audio_data, (0, self.samples_per_frame - len(audio_data)), mode='constant')
        #         else:
        #             audio_data = audio_data[:self.samples_per_frame]
        #
        #     # Construct frame correctly
        #     frame = AudioFrame.from_ndarray(
        #         audio_data.reshape(1, -1),
        #         format="s16",
        #         layout="mono"
        #     )
        #     frame.sample_rate = self.sample_rate  # must be 48000
        #     frame.pts = self.frame_count * self.samples_per_frame
        #     frame.time_base = fractions.Fraction(1, self.sample_rate)
        #     self.frame_count += 1
        #     return frame
        #
        # except Exception as e:
        #     logger.error(f"Audio recv error: {e}")
        #     frame = AudioFrame.from_ndarray(
        #         np.zeros((1, self.samples_per_frame), dtype=np.int16),
        #         format="s16",
        #         layout="mono"
        #     )
        #     frame.sample_rate = self.sample_rate
        #     frame.pts = self.frame_count * self.samples_per_frame
        #     frame.time_base = fractions.Fraction(1, self.sample_rate)
        #     self.frame_count += 1
        #     return frame

    async def add_audio(self, pts, audio_data):
        if not self._is_active:
            return

        try:
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)

            for i in range(0, len(audio_data), self.samples_per_frame):
                chunk = audio_data[i:i + self.samples_per_frame]

                # Calculate pts for this specific chunk
                chunk_pts = pts + (i / self.sample_rate)
                item_to_queue = (chunk_pts, chunk)  # <<< CREATE TUPLE

                try:
                    # self.audio_queue.put_nowait(chunk)
                    self.audio_queue.put_nowait(item_to_queue)  # <<< QUEUE THE TUPLE
                except asyncio.QueueFull:
                    logger.warning("Audio queue full, dropping oldest chunk")
                    try:
                        self.audio_queue.get_nowait()
                        # self.audio_queue.put_nowait(chunk)
                        self.audio_queue.put_nowait(item_to_queue)  # <<< QUEUE THE TUPLE
                    except asyncio.QueueEmpty:
                        pass
        except Exception as e:
            logger.error(f"Error adding audio to queue: {e}")

    def stop(self):
        self._is_active = False


class StreamManager:
    """Manages multiple streaming sessions"""

    def __init__(self, inference_server_url="wss://ftwoltm5pcoc72-8000.proxy.runpod.net/inference"):
        self.sessions: Dict[str, Dict] = {}  # session_id -> {video_track, audio_track, clients}
        self.pcs: Set[RTCPeerConnection] = set()
        self.inference_server_url = inference_server_url
        self.inference_ws = None
        self.reconnect_task = None

    async def start(self):
        """Start the stream manager"""
        self.reconnect_task = asyncio.create_task(self._maintain_inference_connection())

    async def stop(self):
        """Stop the stream manager"""
        if self.reconnect_task:
            self.reconnect_task.cancel()
        if self.inference_ws:
            await self.inference_ws.close()
        # Close all peer connections
        for pc in list(self.pcs):
            await pc.close()
        self.pcs.clear()

    # In main.py, inside the StreamManager class, in _maintain_inference_connection method
    async def _maintain_inference_connection(self):
        """Maintain connection to inference server with reconnection logic"""
        while True:
            try:
                logger.info(f"Connecting to inference server at {self.inference_server_url}")
                # ADD subprotocols argument here
                async with websockets.connect(
                        self.inference_server_url,
                        subprotocols=["webrtc-stream"],
                        max_size=10 * 1024 * 1024  # Set to 2MB, adjust as needed
                ) as websocket:
                    self.inference_ws = websocket
                    logger.info("Connected to inference server")

                    async for message in websocket:
                        if isinstance(message, bytes):
                            try:
                                header_data, raw = message.split(b'\n', 1)
                                header = json.loads(header_data.decode('utf-8'))

                                # Extract the pts from the header
                                pts = header.get('pts')  # <<< ADD THIS LINE
                                if pts is None:
                                    logger.info("Custom: No pts provided. Continue")
                                    continue

                                if header['type'] == 'video_frame':
                                    frame_data = np.frombuffer(raw, dtype=np.uint8).reshape(
                                        header['height'], header['width'], header['channels']
                                    )
                                    if header['session_id'] in self.sessions:
                                        # await self.sessions[header['session_id']]['video_track'].add_frame(frame_data)
                                        # Pass the pts along with the frame data
                                        await self.sessions[header['session_id']]['video_track'].add_frame(pts, frame_data)  # <<< UPDATE THIS LINE
                            except Exception as e:
                                logger.error(f"Binary frame parse failed: {e}")
                        else:
                            # Fallback for JSON-only messages like session_start
                            data = json.loads(message)
                            await self._handle_inference_data(data)

            except Exception as e:
                logger.error(f"Inference server connection error: {e}")
                self.inference_ws = None
                await asyncio.sleep(5)  # Wait before reconnecting

    async def _handle_inference_data(self, data):
        """Handle data received from inference server"""
        try:
            msg_type = data.get('type')
            session_id = data.get('session_id', 'default')
            # Extract the Presentation Timestamp (pts)
            pts = data.get('pts')

            # Ensure we have a pts value before proceeding
            if pts is None:
                logger.warning(f"Message type {msg_type} received without a 'pts'. Skipping.")
                return

            if msg_type == 'video_frame':
                # This logic is for the binary message format, let's update it in _maintain_inference_connection
                pass

            # if msg_type == 'video_frame':
            #     frame_data = np.frombuffer(
            #         bytes.fromhex(data['frame_data']),
            #         dtype=np.uint8
            #     ).reshape(data['height'], data['width'], data['channels'])
            #
            #     if session_id in self.sessions:
            #         await self.sessions[session_id]['video_track'].add_frame(frame_data)

            elif msg_type == 'audio_chunk':
                audio_data = np.frombuffer(
                    bytes.fromhex(data['audio_data']),
                    dtype=np.int16
                )

                if session_id in self.sessions:
                    # await self.sessions[session_id]['audio_track'].add_audio(audio_data)
                    await self.sessions[session_id]['audio_track'].add_audio(pts, audio_data)  # <<< UPDATE THIS LINE

            elif msg_type == 'session_start':
                await self._create_session(session_id, data.get('fps', 25))

            elif msg_type == 'session_end':
                await self._end_session(session_id)

        except Exception as e:
            logger.error(f"Error handling inference data: {e}")

    async def _create_session(self, session_id, fps=25):
        """Create a new streaming session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'video_track': StreamingVideoTrack(fps=fps),
                'audio_track': StreamingAudioTrack(),
                'clients': set()
            }
            logger.info(f"Created streaming session: {session_id}")

    async def _end_session(self, session_id):
        """End a streaming session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['video_track'].stop()
            session['audio_track'].stop()
            del self.sessions[session_id]
            logger.info(f"Ended streaming session: {session_id}")

    async def add_client(self, session_id, pc):
        """Add a client to a streaming session"""
        if session_id not in self.sessions:
            await self._create_session(session_id)

        session = self.sessions[session_id]
        session['clients'].add(pc)
        self.pcs.add(pc)

        # Add tracks to peer connection
        pc.addTrack(session['video_track'])
        pc.addTrack(session['audio_track'])

        logger.info(f"Added client to session {session_id}")

    async def remove_client(self, session_id, pc):
        """Remove a client from a streaming session"""
        if session_id in self.sessions:
            self.sessions[session_id]['clients'].discard(pc)
        self.pcs.discard(pc)
        logger.info(f"Removed client from session {session_id}")

    async def request_inference(self, session_id, audio_file_path):
        """Request inference from the inference server"""
        if self.inference_ws:
            try:
                request = {
                    'type': 'inference_request',
                    'session_id': session_id,
                    'audio_file_path': audio_file_path
                }
                await self.inference_ws.send(json.dumps(request))
                logger.info(f"Sent inference request for session {session_id}")
            except Exception as e:
                logger.error(f"Error sending inference request: {e}")

    async def generate_audio_from_text(self, text, session_id):
        """Generate audio from text using ElevenLabs API"""
        try:
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }

            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            # Make request to ElevenLabs API
            response = requests.post(
                f"{ELEVENLABS_URL}/{ELEVENLABS_VOICE_ID}",
                json=data,
                headers=headers
            )

            if response.status_code == 200:
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"1_{timestamp}.mp3"
                filepath = os.path.join(AUDIO_FOLDER, filename)

                # Save audio file
                with open(filepath, "wb") as f:
                    f.write(response.content)

                logger.info(f"Audio saved: {filepath}")

                # Send inference request to inference server
                await self.request_inference(session_id, filepath)

                return filepath
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error generating audio from text: {e}")
            return None


# Global stream manager
stream_manager = None


async def websocket_handler(request):
    """WebSocket handler for WebRTC signaling"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("WebSocket connection established")

    session_id = request.query.get('session_id', 'default')

    pc = RTCPeerConnection(
        configuration=RTCConfiguration([RTCIceServer("stun:stun.l.google.com:19302")])
    )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"PeerConnection state: {pc.connectionState}")
        if pc.connectionState == "closed":
            await stream_manager.remove_client(session_id, pc)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await ws.send_str(json.dumps({
                "type": "ice-candidate",
                "candidate": {
                    "candidate": candidate.candidate,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }
            }))

    # Add client to stream manager
    await stream_manager.add_client(session_id, pc)

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)

                if data["type"] == "offer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type=data["type"]))

                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)

                    await ws.send_str(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    }))



                elif data["type"] == "ice-candidate":
                    candidate_info = data["candidate"]
                    # Check if candidate is not None/empty (end-of-candidates has null candidate)
                    if candidate_info.get("candidate"):
                        try:
                            # Parse the SDP candidate string
                            candidate_str = candidate_info["candidate"]
                            parts = candidate_str.split()
                            # Extract required fields from SDP candidate string
                            # Format: candidate:<foundation> <component> <protocol> <priority> <ip> <port> typ <type> ...
                            if len(parts) >= 8:
                                foundation = parts[0].split(':')[1]  # Remove 'candidate:' prefix
                                component = int(parts[1])
                                protocol = parts[2].lower()
                                priority = int(parts[3])
                                ip = parts[4]
                                port = int(parts[5])
                                typ = parts[7]  # parts[6] is 'typ'
                                ice_candidate = RTCIceCandidate(
                                    foundation=foundation,
                                    component=component,
                                    protocol=protocol,
                                    priority=priority,
                                    ip=ip,
                                    port=port,
                                    type=typ
                                )
                                ice_candidate.sdpMid = candidate_info.get("sdpMid")
                                ice_candidate.sdpMLineIndex = candidate_info.get("sdpMLineIndex")
                                await pc.addIceCandidate(ice_candidate)
                                logger.debug(f"Added ICE candidate: {typ} {ip}:{port}")
                            else:
                                logger.error(f"Invalid candidate format: {candidate_str}")
                        except Exception as e:
                            logger.error(f"Failed to add ICE candidate: {e}")
                            logger.error(f"Candidate data: {candidate_info}")
                    else:
                        logger.debug("Received end-of-candidates signal")

                elif data["type"] == "request_inference":
                    # Client requests inference for an audio file
                    audio_file_path = data.get("audio_file_path")
                    if audio_file_path:
                        await stream_manager.request_inference(session_id, audio_file_path)

                elif data["type"] == "text_to_speech":
                    # Client sends text for TTS conversion
                    text = data.get("text", "").strip()
                    if text:
                        logger.info(f"Received TTS request: {text}")
                        await stream_manager.generate_audio_from_text(text, session_id)
                        await ws.send_str(json.dumps({
                            "type": "tts_status",
                            "message": "Processing text to speech..."
                        }))
                    else:
                        await ws.send_str(json.dumps({
                            "type": "tts_error",
                            "message": "No text provided"
                        }))

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

        elif msg.type == WSMsgType.ERROR:
            logger.error(f"WebSocket closed with error: {ws.exception()}")

    if pc.connectionState != "closed":
        await pc.close()
    return ws


async def index(request):
    """Serve the main HTML page"""
    content = """
<!DOCTYPE html>
<html>
<head>
    <title>MuseTalk WebRTC Stream</title>
    <style>
        body { 
            background-color: #222; 
            color: white; 
            font-family: 'Inter', sans-serif; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            padding: 20px; 
            margin: 0;
            min-height: 100vh;
            box-sizing: border-box;
        }
        h1 { color: #007bff; margin-bottom: 30px; }
        #video-container {
            position: relative;
            width: 80%; 
            max-width: 800px; 
            margin-bottom: 20px;
            border: 2px solid #555; 
            border-radius: 12px; 
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        video { 
            width: 100%; 
            height: auto; 
            display: block;
            border-radius: 10px;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        button, input[type="file"] {
            padding: 12px 28px;
            font-size: 17px;
            font-weight: bold;
            background-image: linear-gradient(to right, #007bff, #0056b3);
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        button:hover { 
            background-image: linear-gradient(to right, #0056b3, #003d80);
            transform: translateY(-2px);
        }
        #status {
            margin: 25px 0;
            padding: 12px 25px;
            background-color: #333;
            border-radius: 8px;
            font-size: 1.1em;
        }
        .session-input {
            margin: 10px;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #555;
            background: #333;
            color: white;
        }
    </style>
</head>
<body>
    <h1>MuseTalk WebRTC Live Stream</h1>
    <div>
        <input type="text" id="sessionId" class="session-input" placeholder="Session ID (default: 'default')" value="default">
    </div>
    <div id="status">Status: Disconnected</div>
    <div id="video-container">
        <video id="remoteVideo" autoplay playsinline controls></video>
    </div>
    <div class="controls">
    <button onclick="startStream()">Start Stream</button>
    <button onclick="stopStream()">Stop Stream</button>
    <input type="file" id="audioFile" accept=".wav,.mp3" style="display: none;">
    <button onclick="document.getElementById('audioFile').click()">Upload Audio</button>
    <div style="margin-top: 15px; width: 100%;">
        <input type="text" id="textInput" placeholder="Enter text for speech synthesis..." 
               style="width: 70%; padding: 12px; margin-right: 10px; border-radius: 5px; border: 1px solid #555; background: #333; color: white; font-size: 16px;">
        <button onclick="sendTextToSpeech()" style="width: 25%;">Generate Speech</button>
    </div>
</div>

    <script>
        let pc = null;
        let ws = null;
        let remoteVideo = document.getElementById('remoteVideo');
        let currentSessionId = 'default';

        document.getElementById('audioFile').addEventListener('change', handleAudioUpload);

        function updateStatus(message) {
            document.getElementById('status').textContent = 'Status: ' + message;
            console.log('Status:', message);
        }

        async function handleAudioUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            // In a real implementation, you'd upload the file to a server
            // For now, we'll just simulate requesting inference
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'request_inference',
                    audio_file_path: file.name  // In practice, this would be a server path
                }));
                updateStatus('Requested inference for: ' + file.name);
            }
        }

        async function startStream() {
            currentSessionId = document.getElementById('sessionId').value || 'default';

            if (pc && pc.connectionState !== 'closed') {
                updateStatus('Already streaming...');
                return;
            }

            updateStatus('Connecting...');

            try {
                ws = new WebSocket(`wss://test-app-adfz.onrender.com/ws?session_id=${currentSessionId}`);

                ws.onopen = function() {
                    updateStatus('WebSocket connected, creating PeerConnection...');
                    createPeerConnection();
                };

                ws.onmessage = async function(event) {
                    const data = JSON.parse(event.data);

                    if (data.type === 'answer') {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                        updateStatus('Streaming...');
                    } else if (data.type === 'ice-candidate') {
                        if (pc && data.candidate) {
                            await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
                        }
                    } else if (data.type === 'tts_status') {
                        updateStatus(data.message);
                    } else if (data.type === 'tts_error') {
                        updateStatus('Error: ' + data.message);
                    }
                };

                ws.onerror = function(error) {
                    updateStatus('WebSocket error');
                    console.error('WebSocket Error:', error);
                };

                ws.onclose = function() {
                    updateStatus('WebSocket disconnected');
                    stopStream();
                };

            } catch (e) {
                updateStatus('Failed to start stream: ' + e.message);
                console.error('Start Stream Error:', e);
            }
        }

        async function createPeerConnection() {
            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            pc.addTransceiver('video', { direction: 'recvonly' });
            pc.addTransceiver('audio', { direction: 'recvonly' });

            pc.onicecandidate = function(event) {
                if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'ice-candidate',
                        candidate: {
                            candidate: event.candidate.candidate,
                            sdpMid: event.candidate.sdpMid,
                            sdpMLineIndex: event.candidate.sdpMLineIndex,
                        }
                    }));
                }
            };

            pc.ontrack = function(event) {
                if (remoteVideo.srcObject !== event.streams[0]) {
                    remoteVideo.srcObject = event.streams[0];
                    
                    // CRITICAL: Set playback rate to normal speed
                    remoteVideo.playbackRate = 1.0;
                    
                    // Remove any buffering delays
                    remoteVideo.setAttribute('autoplay', true);
                    remoteVideo.setAttribute('playsinline', true);
                    
                    remoteVideo.play().catch(e => console.error("Error playing video:", e));
                    updateStatus('Stream received');
                }
            };

            pc.onconnectionstatechange = function() {
                updateStatus('WebRTC: ' + pc.connectionState);
                if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
                    stopStream();
                }
            };

            try {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'offer',
                        sdp: offer.sdp
                    }));
                }
            } catch (e) {
                console.error('Error creating offer:', e);
                stopStream();
            }
        }

        function stopStream() {
            if (pc) {
                pc.close();
                pc = null;
            }
            if (ws) {
                ws.close();
                ws = null;
            }
            if (remoteVideo.srcObject) {
                remoteVideo.srcObject.getTracks().forEach(track => track.stop());
                remoteVideo.srcObject = null;
            }
            updateStatus('Disconnected');
        }
        
        function sendTextToSpeech() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        updateStatus('Please enter some text');
        return;
    }
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        updateStatus('Not connected to server');
        return;
    }
    
    updateStatus('Generating speech from text...');
    
    ws.send(JSON.stringify({
        type: 'text_to_speech',
        text: text
    }));
    
    // Clear the input
    textInput.value = '';
}

// Add Enter key support for text input
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendTextToSpeech();
            }
        });
    }
});

        window.addEventListener('beforeunload', stopStream);
    </script>
</body>
</html>
    """
    return web.Response(text=content, content_type='text/html')


async def init_app():
    """Initialize the aiohttp web application"""
    app = web.Application()

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket_handler)

    for route in list(app.router.routes()):
        cors.add(route)

    return app


async def main():
    """Main function"""
    global stream_manager

    # Initialize stream manager
    stream_manager = StreamManager("ws://ftwoltm5pcoc72-8000.proxy.runpod.net/inference")  # Inference server URL
    await stream_manager.start()

    # Create and start web app
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    logger.info("WebRTC Streaming Server started on http://0.0.0.0:8080")
    logger.info("Connecting to inference server at ws://localhost:8000/inference")

    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await stream_manager.stop()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
