# リポジトリ分析

## 概要
```markdown
## プロジェクト概要

このPythonプロジェクトは、オンライン会議プラットフォーム（LiveKit）で自動化されたバーチャルキャラクターボット「Pipecat Bot」を提供することを目的としています。各会議室のライフサイクルを管理し、テキストチャットや音声処理機能を備えた対話型エージェントサービスを実現します。

## 主な目的

Pipecat Botの主な目的は、オンライン会議中にユーザーとの対話を支援することです。これには、音声認識（STT）と音声合成（TTS）、および大規模言語モデル（LLM）を使用して、各会議室でのユーザー体験を向上させることが含まれます。

## 主要なコンポーネントとその役割

- **QueueConsumer**: SQSキューを監視し、新しい会議室の要求に応じてBotを生成・管理する役割を担います。
- **Bot**: 各会議室内での対話ロジックを担当し、音声およびテキストのやり取りを処理します。音声入力/出力の有無に応じて音声処理パイプラインを設定します。
- **DatabaseService**: Supabaseを通じて会議室やキャラクターの設定情報を取得するためのサービスです。
- **Event Handlers (LiveKitEventHandlersMixin)**: イベント駆動で参加者のジョインやデータ受信、退出時のロジックを扱います。
- **LLMFrameStore**: 生成された大規模言語モデルのコンテキストをデータベースに保存します。

## 使用している主要な外部APIやライブラリ

- **LiveKit API**: オンライン会議プラットフォームの音声・テキストチャットを行います。
- **Supabase**: データストレージとして使用され、会議室やキャラクターの情報を管理します。
- **AWS SQS**: 会議室のアクティベーションと管理にキューを使用します。
- **Gladia API (STT)**: 音声チャットのテキスト化を実現します。
- **Cartesia API (TTS)**: テキストチャットの音声化を担当します。
- **Pipecat**: 音声処理パイプライン構築のためのライブラリ。
- **OpenAI LLM**: チャット内容の生成や文脈の管理に利用されます。

## アーキテクチャの特徴

- **非同期処理**: asyncioを活用して効率的な非同期タスク実行を実現しています。特に、QueueConsumerやBotの処理がこれに依存しています。
- **マイクロサービス構造**: 各サービス（Bot、キュー監視、音声処理など）は独立しており、責務分離が明確です。
- **イベント駆動型アプリケーション**: イベントハンドラによりユーザの参加に応じて動的に処理を進めます。
- **設定の外部化**: dotenvを使った設定管理を行い、APIキーやURLなどの機密情報を環境変数で管理しています。

このプロジェクトは、今後さらに多機能なオンライン会議ボットを開発するための基盤を提供しています。
```

## ディレクトリ構造
├── .Dockerignore
├── agent/
│   ├── config/
│   │   └── settings.py
│   ├── core/
│   │   ├── bot.py
│   │   └── queue_consumer.py
│   ├── handlers/
│   │   └── events.py
│   ├── main.py
│   ├── processors/
│   │   ├── chat.py
│   │   └── llm_frame_store.py
│   ├── services/
│   │   └── DatabaseService.py
│   └── utils.py
├── Dockerfile
├── processors/
├── README.md
└── requirements.txt

## ソースコード
### Dockerfile
```Dockerfile
FROM python:3.10-slim-bookworm

RUN useradd -m debian

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY /agent /app/agent

USER debian

ENV ROOM_ID=
ENV AUDIO_ENABLED=

CMD ["python", "/app/agent/main.py"]

```

### README.md
```md
## Install / Run

Please use a virtual environment to install the dependencies.

```python
pip install -r requirements.txt
python agent/main.py
```

## Architecture

The worker listens to a queue for new rooms to process, and handles the lifecycle of each room.

In this image, this repo is the Pipecat Bot is located in this repo.

![image](https://github.com/user-attachments/assets/adeabed4-b5db-49da-93f3-ac3f310afa1d)

```

### agent\main.py
```py
# main.py
import asyncio
import sys

from config.settings import Settings
from core.queue_consumer import QueueConsumer
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    logger.info("Starting queue consumer")
    consumer = QueueConsumer()
    await consumer.start()


if __name__ == "__main__":
    asyncio.run(main())

```

### agent\utils.py
```py
from config.settings import Settings
from livekit import api
from supabase import Client, create_client


def generate_livekit_token(room_name: str, participant_name: str) -> str:
    api_key = Settings.LIVEKIT_API_KEY
    api_secret = Settings.LIVEKIT_API_SECRET

    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )

    return token.to_jwt()


def create_supabase_client() -> Client:
    supabase: Client = create_client(
        Settings.SUPABASE_URL,
        Settings.SUPABASE_SERVICE_ROLE
    )
    return supabase

```

### agent\config\settings.py
```py
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LiveKit Settings
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", None)
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", None)
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", None)
    ROOM_ID = os.getenv("ROOM_ID", None)

    # Database Settings
    SUPABASE_URL = os.getenv("SUPABASE_URL", None)
    SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE", None)

    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
    GLADIA_STT_API_KEY = os.getenv("GLADIA_STT_API_KEY", None)
    CARTESIA_TTS_API_KEY = os.getenv("CARTESIA_TTS_API_KEY", None)

    # Audio Settings
    AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "False").lower() == "true"
    TTS_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"  # British Lady

    # LLM Settings
    LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
    LLM_BASE_URL = "https://openrouter.ai/api/v1"

    # User Idle Settings
    IDLE_TIMEOUT = 30.0

    # Queue Settings
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/210309033471/bunny-ai-agent.fifo")
    MAX_CONCURRENT_ROOMS = int(os.getenv("MAX_CONCURRENT_ROOMS", "10"))

```

### agent\core\bot.py
```py
from config.settings import Settings
from handlers.events import LiveKitEventHandlersMixin
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.logger import FrameLogger
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.gladia import GladiaSTTService, Language
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport
from processors.chat import LivekitChatForwarder
from processors.llm_frame_store import LLMFrameStore
from services.DatabaseService import DatabaseService
from utils import create_supabase_client, generate_livekit_token


class Bot(LiveKitEventHandlersMixin):
    def __init__(self, room_id, audio_enabled):
        self.room_id = room_id
        self.audio_enabled = audio_enabled
        try:
            self.db_client = create_supabase_client()
            self.db_service = DatabaseService(self.db_client)
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise RuntimeError("Failed to initialize database connection") from e

    async def run(self):

        # Get configuration
        avatar_id = await self.db_service.get_room_configuration(self.room_id)
        self.name, prompt = await self.db_service.get_character_configuration(avatar_id)

        # Setup and run pipeline
        pipeline_components = await self._setup_pipeline_components(prompt)
        self.task = PipelineTask(Pipeline(pipeline_components))
        self.setup_event_handlers()
        self.runner = PipelineRunner()
        await self.runner.run(self.task)

    async def _setup_pipeline_components(self, prompt):
        # Setup transport
        self.transport = LiveKitTransport(
            url=Settings.LIVEKIT_URL,
            token=generate_livekit_token(self.room_id, self.name),
            room_name=self.room_id,
            params=LiveKitParams(
                audio_out_enabled=self.audio_enabled,
                audio_in_enabled=self.audio_enabled,
                vad_enabled=self.audio_enabled,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.1)) if self.audio_enabled else None,
            ),
        )

        # Setup core components
        idle_check = UserIdleProcessor(callback=self.idle_disconnect,
                                       timeout=Settings.IDLE_TIMEOUT)

        llm = OpenAILLMService(api_key=Settings.OPENROUTER_API_KEY,
                               model=Settings.LLM_MODEL,
                               base_url=Settings.LLM_BASE_URL)

        # Must be duped to prevent recursion error in pipeline
        stt_chat = LivekitChatForwarder(self.transport)
        llm_chat = LivekitChatForwarder(self.transport)

        self.context = OpenAILLMContext([
            {"role": "system", "content": prompt},
        ])
        context_aggregator = llm.create_context_aggregator(self.context)
        context_store = LLMFrameStore(self.db_client)
        frame_logger = FrameLogger()

        if self.audio_enabled:
            stt = GladiaSTTService(
                api_key=Settings.GLADIA_STT_API_KEY,
                confidence=0.7,
                params=GladiaSTTService.InputParams(
                    language=Language.EN,
                    audio_enhancer=True,
                    sample_rate=16000
                )
            )

            tts = CartesiaTTSService(
                api_key=Settings.CARTESIA_TTS_API_KEY,
                voice_id=Settings.TTS_VOICE_ID,
            )

            return [
                self.transport.input(),
                idle_check,
                stt,
                stt_chat,
                context_aggregator.user(),
                llm,
                llm_chat,
                tts,
                self.transport.output(),
                context_aggregator.assistant(),
                frame_logger,
                context_store
            ]
        else:
            return [
                self.transport.input(),
                idle_check,
                context_aggregator.user(),
                llm,
                llm_chat,
                context_aggregator.assistant(),
                frame_logger,
                context_store
            ]


    async def cleanup(self):
        # TODO: update to the latest pipecat (when released) to fix the disconnect error messages
        logger.debug(f"Cleaning up room {self.room_id}")
        try:
            await self.task.queue_frame([EndFrame()])
            await self.runner.stop_when_done()
        except Exception as e:
            logger.error(f"Error cleaning up room {self.room_id}: {e}")


    def setup_event_handlers(self):
        async def handle_first_participant(_, participant_id):
            await self.on_first_participant_joined(participant_id)

        async def handle_data_received(_, data, participant_id):
            await self.on_data_received(data, participant_id)

        async def handle_participant_left(_, participant_id, reason):
            await self.on_participant_left(participant_id, reason)

        self.transport.event_handler("on_first_participant_joined")(handle_first_participant)
        self.transport.event_handler("on_data_received")(handle_data_received)
        self.transport.event_handler("on_participant_left")(handle_participant_left)

```

### agent\core\queue_consumer.py
```py
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import boto3
from config.settings import Settings
from loguru import logger

from .bot import Bot


class QueueConsumer:
    def __init__(self, queue_url=None):
        """Initialize the QueueConsumer.

        Args:
            queue_url: Optional SQS queue URL. If not provided, uses Settings.SQS_QUEUE_URL
        """
        self.queue_url = queue_url or Settings.SQS_QUEUE_URL
        if not self.queue_url:
            raise ValueError("Queue URL must be provided either directly or via Settings.SQS_QUEUE_URL")

        self.sqs = boto3.client('sqs',
                                endpoint_url=Settings.SQS_QUEUE_URL,
                                region_name=Settings.AWS_REGION
                                )
        self.executor = ThreadPoolExecutor(max_workers=Settings.MAX_CONCURRENT_ROOMS)
        self.active_rooms: Dict[str, Bot] = {}  # room_id -> Bot mapping
        self._running = False
        self._task = None

    async def receive_message(self) -> Optional[dict]:
        """Receive a message from the queue."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20
                )
            )
            messages = response.get('Messages', [])
            return messages[0] if messages else None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def delete_message(self, receipt_handle: str):
        """Delete a message from the queue."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.sqs.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            return False

    async def process_room(self, room_id: str):
        """Process a single room."""
        try:
            bot = Bot(room_id, audio_enabled=False)
            self.active_rooms[room_id] = bot
            logger.info(f"Created bot for room {room_id}, current room count: {len(self.active_rooms)}")
            await bot.run()
        except Exception as e:
            logger.error(f"Error processing room {room_id}: {e}")
        finally:
            if room_id in self.active_rooms:
                bot = self.active_rooms[room_id]
                await bot.cleanup()
                self.active_rooms.pop(room_id, None)
                logger.debug(f"Cleaned up room {room_id}, current room count: {len(self.active_rooms)}")

    async def start(self):
        """Start consuming messages from the queue."""
        self._running = True
        while self._running:
            if len(self.active_rooms) >= Settings.MAX_CONCURRENT_ROOMS:
                await asyncio.sleep(5)
                continue

            message = await self.receive_message()
            if not message:
                await asyncio.sleep(1)  # Avoid tight loop when no messages
                continue

            try:
                body = json.loads(message['Body'])
                room_id = body['roomName']
                receipt_handle = message['ReceiptHandle']
                logger.info(f"Received message for room: {room_id}")

                # Start room processing
                asyncio.create_task(self.process_room(room_id))

                # Delete message after starting processing
                await self.delete_message(receipt_handle)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

    async def stop(self):
        """Stop consuming messages."""
        logger.info("Stopping queue consumer")
        self._running = False

```

### agent\handlers\events.py
```py
# handlers/events.py
import json
from typing import Protocol

from livekit import rtc
from loguru import logger
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.openai import OpenAILLMContext
from pipecat.transports.services.livekit import LiveKitTransport


class BotProtocol(Protocol):
    """ Interface for Bot """
    transport: LiveKitTransport
    task: PipelineTask
    runner: PipelineRunner
    context: OpenAILLMContext
    name: str

    async def cleanup(self) -> None:
        ...


class LiveKitEventHandlersMixin:
    async def on_first_participant_joined(self: BotProtocol, participant_id: str):
        logger.info(f"Participant joined: {participant_id}")
        await rtc.ChatManager(self.transport._client._room).send_message(f"{self.name} has joined the chat!")

    async def on_data_received(self: BotProtocol, data, participant_id: str):
        try:
            try:
                decoded_data = data.decode('utf-8')
            except UnicodeDecodeError as e:
                logger.error(f"Failed to decode UTF-8 data from participant {participant_id}: {e}")
                return

            try:
                incoming = rtc.ChatMessage.from_jsondict(json.loads(decoded_data))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from participant {participant_id}: {e}")
                return

            messages = self.context.get_messages()
            messages.append({
                "role": "user",
                "content": incoming.message,
            })

            await self.task.queue_frame(LLMMessagesFrame(messages))

        except Exception as e:
            logger.error(f"Error processing incoming message from {participant_id}: {e}")

    async def on_participant_left(self: BotProtocol, participant_id: str, reason: str):
        logger.info(f"Room ending because {participant_id} left")
        await self.cleanup()

    async def idle_disconnect(self: BotProtocol, _: UserIdleProcessor):
        # TODO: better chat protocol
        await rtc.ChatManager(self.transport._client._room).send_message("[SYSTEM] User idle - Disconnecting...")
        await self.cleanup()

```

### agent\processors\chat.py
```py
import re

from livekit import rtc
from pipecat.frames.frames import (BotStoppedSpeakingFrame, EndFrame, Frame,
                                   TextFrame, TranscriptionFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.services.livekit import LiveKitTransport
from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter


class LivekitChatForwarder(FrameProcessor):
    """Sends text content to the Livekit chat room, aggregating sentences if needed before sending."""

    def __init__(self, transport: LiveKitTransport):
        super().__init__()
        self.transport = transport
        self._aggregation = ""
        self._text_filter = MarkdownTextFilter(
            params=MarkdownTextFilter.InputParams(
                enable_text_filter=True,
                filter_code=True,
                filter_tables=True
            )
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame and forwards its content to the Livekit chat room."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            frame.text = await self._process_text(frame.text)
            await rtc.ChatManager(self.transport._client._room).send_message("[TRANSCRIPTION]:" + frame.text)

        elif isinstance(frame, TextFrame):
            frame.text = await self._process_text(frame.text)
            self._aggregation += frame.text

            if match_endofsentence(self._aggregation):
                await rtc.ChatManager(self.transport._client._room).send_message(self._aggregation)
                self._aggregation = ""
        elif isinstance(frame, EndFrame) or isinstance(frame, BotStoppedSpeakingFrame):
            if self._aggregation:
                await rtc.ChatManager(self.transport._client._room).send_message(self._aggregation)
                self._aggregation = ""

        await self.push_frame(frame, direction)

    async def _process_text(self, text: str):

        # https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/utils/text/markdown_text_filter.py#L41
        text = self._text_filter.filter(text)
        text = re.sub(r'[@#$%^&*]', '', text)
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # remove emojis
        return text

```

### agent\processors\llm_frame_store.py
```py

from pipecat.frames.frames import Frame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai import OpenAILLMContextFrame
from supabase import Client


class LLMFrameStore(FrameProcessor):
    """ Save context to Supabase """

    def __init__(self, supabase: Client):
        super().__init__()
        self.supabase = supabase

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame and forwards its content to the Livekit chat room."""
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            await self._save_context(frame.context)

        await self.push_frame(frame, direction)

    async def _save_context(self, context: OpenAILLMContext):
        messages = context.get_messages()
        # TODO: save to supabase

```

### agent\services\DatabaseService.py
```py
from loguru import logger


class DatabaseService:
    def __init__(self, supabase_client):
        self.client = supabase_client

    async def get_room_configuration(self, room_id):
        try:
            response = self.client.table("sessions").select("*").eq("id", room_id).single().execute()
            return response.data["character_id"]
        except Exception as e:
            logger.error(f"Error fetching room configuration: {e}")
            raise

    async def get_character_configuration(self, character_id):
        try:
            response = self.client.table("characters").select("*").eq("id", character_id).single().execute()
            return response.data["name"], response.data["description"]
        except Exception as e:
            logger.error(f"Error fetching character configuration: {e}")
            raise

```
 
