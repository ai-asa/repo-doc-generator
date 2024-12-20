# リポジトリ分析

## 概要
# プロジェクト概要

このプロジェクトは、音声合成（TTS: Text-to-Speech）とチャット機能を提供するサーバーアプリケーションを構築することを目的としています。主に、外部のTTSサービスを利用してテキストを音声に変換し、また、AIを利用したチャットの応答を生成します。

## 主な目的

本プロジェクトの主な目的は、複数のTTSプロバイダーを利用した音声合成機能とAIを用いたチャット機能を提供するサーバーを構築することです。これにより、異なる音声サービスプロバイダーからのTTS機能を統合し、ユーザーに選択肢を提供します。

## 主要なコンポーネントとその役割

- **APIサーバー（src/index.ts）**:
  - Honoフレームワークを用いてAPIエンドポイントを設定し、CORSの設定などを行います。
  - `/api/chat` エンドポイントでチャット機能を提供します。
  - `/api/tts` エンドポイントでTTS機能を提供します。

- **チャットモジュール（src/routes/chat.ts）**:
  - `generateText`を用いてAIモデルを用いたテキストの生成を行います。
  - ユーザーから送信されたメッセージを処理し、AIによる応答を生成して返します。

- **TTSモジュール（src/routes/tts.ts と tts/app内のPythonスクリプト群）**:
  - クライアントリクエストに基づいて選択されたTTSサービス（OpenAI, Azure, ElevenLabs）にテキストを送り、音声データを取得します。
  - PythonのFastAPIを用いてTTS処理を担当します。Python側で各TTSサービスへの接続を管理します。

- **スキーマ（src/schemaディレクトリ）**:
  - APIのリクエストおよびレスポンスを定義し、バリデーションを行うためのZodスキーマを提供します。

- **Pythonサービス（tts/app/servicesディレクトリ）**:
  - 各TTSプロバイダーに対するサービスクラスを提供し、TTSリクエストを処理します。

## 使用している主要な外部APIやライブラリ

- **HonoとZod**:
  - APIサーバーのルーティングやスキーマバリデーションに使用しています。
- **AIライブラリ**:
  - `generateText`関数により、AIを活用したチャット応答を生成します。
- **FastAPI**:
  - Pythonサーバーとしての役割を担い、TTSリクエストを処理します。
- **pipecat**:
  - 各TTSサービスの接続および操作に用いるライブラリとして使用されています（推察）。

## アーキテクチャの特徴

- サーバーサイドはTypeScriptを用いたAPIサーバーと、Pythonを用いたTTSサーバーが非同期に動作します。
- フロントエンド（UI部分は不明）から送信されたリクエストに基づき、必要な処理をバックエンドで分担しています。
- Honoフレームワークを利用して軽量なマイクロサービス的APIサーバーを構築し、柔軟なスキーマバリデーションを実現しています。
- (.envファイルを用いて）APIキーや外部サービスURLを環境変数で管理することで、異なる環境での簡単な設定変更を可能にしています。

このように、本プロジェクトはモジュール化と拡張性に優れた設計を特徴としており、異なる音声合成プロバイダーを柔軟に利用することができます。

## ディレクトリ構造
├── bun.lockb
├── docs/
│   └── provider-verification/
├── package.json
├── README.md
├── src/
│   ├── index.ts
│   ├── routes/
│   │   ├── chat.ts
│   │   └── tts.ts
│   └── schema/
│       ├── error.ts
│       └── messages.ts
├── tsconfig.json
└── tts/
    ├── app/
    │   ├── main.py
    │   ├── routes/
    │   │   └── tts.py
    │   ├── schema/
    │   │   └── tts_request.py
    │   └── services/
    │       ├── azure_service.py
    │       ├── base.py
    │       ├── elevenlabs_service.py
    │       └── openai_service.py
    └── requirements.txt

## ソースコード
### package.json
```json
{
  "name": "bunny-ai-chat-worker",
  "scripts": {
    "dev": "bun run dev:chat",
    "dev:all": "concurrently \"bun run dev:chat\" \"bun run dev:tts\"",
    "dev:chat": "bun run --port 3000 --hot src/index.ts",
    "dev:tts": "uvicorn tts.app.main:app --host 0.0.0.0 --port 5000 --reload --env-file tts/.env"
  },
  "dependencies": {
    "@hono/zod-openapi": "^0.18.3",
    "@hono/zod-validator": "^0.4.1",
    "@openrouter/ai-sdk-provider": "^0.0.6",
    "@scalar/hono-api-reference": "^0.5.163",
    "ai": "^4.0.13",
    "hono": "^4.6.13",
    "zod": "^3.23.8"
  },
  "devDependencies": {
    "@types/bun": "latest",
    "concurrently": "latest"
  }
}

```

### README.md
```md

To install dependencies:
```sh
bun install
```

To run:
```sh
bun run dev
```

## Pipecat TTS Server

To run the Pipecat TTS server:
```sh
pip install -r tts/requirements.txt
```

```sh
bun run dev:tts
```

Open [http://localhost:3001/reference](http://localhost:3001/reference)

## Run Both Servers:

```sh
bun run dev:all
```

```

### tsconfig.json
```json
{
  "compilerOptions": {
    "strict": true,
    "jsx": "react-jsx",
    "jsxImportSource": "hono/jsx",
    "paths": {
      "@/*": [
        "./src/*"
      ]
    }
  }
}
```

### src\index.ts
```ts
import chatRoute from '@/routes/chat';
import ttsRoute from '@/routes/tts';
import { OpenAPIHono } from '@hono/zod-openapi';
import { apiReference } from '@scalar/hono-api-reference';
import { cors } from 'hono/cors';

const app = new OpenAPIHono()



app.use('/api/*', cors(
    {
        origin: '*', // TODO: change later
    },
));

app.route('/api/chat', chatRoute);
app.route('/api/tts', ttsRoute);

app.doc('/doc', {
    openapi: '3.0.0',
    info: {
        version: '1.0.0',
        title: 'API',
    },
})

app.get(
    '/reference',
    apiReference({
        spec: {
            url: '/doc',
        },
    }),
)

export default app;
```

### src\routes\chat.ts
```ts
import { genericError } from "@/schema/error";
import { MessagesSchema } from "@/schema/messages";
import { OpenAPIHono, createRoute, z } from "@hono/zod-openapi";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { generateText } from 'ai';

export const app = new OpenAPIHono();

const chatRoute = createRoute({
    method: "post",
    path: "",
    request: {
        body: {
            content: {
                "application/json": {
                    schema: MessagesSchema
                },
            },
        },
        required: true
    },
    responses: {
        200: {
            content: {
                "application/json": {
                    schema: z.object({
                        reply: z.string(),
                    }),
                },
            },
            description: "Success",
        },

        400: genericError,
        500: genericError
    },
});

app.openapi(chatRoute, async (c) => {
    const openrouter = createOpenRouter({
        apiKey: process.env.OPENROUTER_API_KEY,
    });

    const { messages } = c.req.valid('json');

    if (messages.length === 0) {
        return c.json(
            { error: "No messages provided" },
            400
        );
    }

    const response = await generateText({
        model: openrouter("meta-llama/llama-3-8b-instruct:free"),
        messages: [
            { role: "system", content: "You are a helpful assistant." },
            ...messages.map((message) => ({
                role: (message.user.id === 1 ? "user" : "assistant") as "user" | "assistant", // TODO: this should not be like this
                content: message.content,
            })),
        ],
    });

    const reply = response.text;

    if (!reply) {
        return c.json(
            { error: "Failed to generate a reply" },
            500
        );
    }

    return c.json({ reply }, 200);
});

export default app;

```

### src\routes\tts.ts
```ts
import { z } from 'zod';
import { OpenAPIHono, createRoute } from '@hono/zod-openapi';

const ttsSchema = z.object({
  text: z.string().min(1, "Text cannot be empty"),
  provider: z.enum(['openai', 'azure', 'elevenlabs'])
});

const ttsResponseSchema = z.object({
    audioData: z.string().optional()
});

const ttsErrorSchema = z.object({
    error: z.string()
});


const app = new OpenAPIHono();

const ttsRoute = createRoute({
    method: 'post',
    path: '',
    request: {
      body: {
        content: {
          'application/json': {
            schema: ttsSchema,
          },
        },
        required: true,
      },
    },
    responses: {
      200: {
        content: {
          'application/json': {
            schema: ttsResponseSchema, 
          },
        },
        description: "Success",
      },
      400: {
        content: {
          'application/json': {
            schema: ttsErrorSchema,
          },
        },
        description: "Bad Request",
      },
      500: {
        content: {
          'application/json': {
            schema: ttsErrorSchema,
          },
        },
        description: "Internal Server Error",
      },
    },
});

app.openapi(ttsRoute, async (c) => {
    const { text, provider } = c.req.valid('json');
    const pipecatServer = process.env.PIPECAT_SERVER_URL || 'http://localhost:5000';
  
    const body = {
        text,
        provider: provider || 'openai'
    };

    // Request to Python(pipecat) server
    const response = await fetch(`${pipecatServer}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  
    if (!response.ok) {
      return c.json({ error: 'Failed to call pipecat server' }, 500);
    }
  
    const data = await response.json();
    if (data.error) {
      return c.json({ error: data.error }, 500);
    }
  
    return c.json({ audioData: data.audioData }, 200);
  });
  
  export default app;

```

### src\schema\error.ts
```ts
import { z } from "@hono/zod-openapi";


const errorSchema = z.object({
    error: z.string(),
})


export const genericError = {
        content: {
            "application/json": {
                schema: errorSchema
            },
        },
        description: "Request failed",
}

```

### src\schema\messages.ts
```ts
import { z } from '@hono/zod-openapi';

export const MessagesSchema = z.object({
    messages: z.array(
        z.object({
            user: z.object({ id: z.number() }),
            content: z.string(),
        }).required()
    ),
});

```

### tts\app\main.py
```py
from fastapi import FastAPI
from dotenv import load_dotenv
from .routes import tts

load_dotenv()

app = FastAPI()

app.include_router(tts.router)

```

### tts\app\routes\tts.py
```py
# routes/tts.py
from fastapi import APIRouter, HTTPException
from ..schema.tts_request import TTSRequest, TTSResponse
from ..services.base import BaseTTSService, ErrorFrame, TTSAudioRawFrame
from ..services.openai_service import MyOpenAITTSService
from ..services.azure_service import MyAzureTTSService
from ..services.elevenlabs_service import MyElevenLabsTTSService
import base64

router = APIRouter()

@router.post("/tts", response_model=TTSResponse)
async def tts_endpoint(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if req.provider == "openai":
        tts_service: BaseTTSService = MyOpenAITTSService()
    elif req.provider == "azure":
        tts_service: BaseTTSService = MyAzureTTSService()
    elif req.provider == "elevenlabs":
        tts_service: BaseTTSService = MyElevenLabsTTSService()
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")

    audio_data = b""

    try:
        async for frame in tts_service.run_tts(text):
            if isinstance(frame, ErrorFrame):
                raise HTTPException(status_code=500, detail=f"TTS error: {frame.error}")
            elif isinstance(frame, TTSAudioRawFrame):
                audio_data += frame.audio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    base64_audio = base64.b64encode(audio_data).decode('utf-8')
    audio_uri = f"data:audio/wav;base64,{base64_audio}"

    return TTSResponse(audioData=audio_uri)

```

### tts\app\schema\tts_request.py
```py
from pydantic import BaseModel, Field, field_validator

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    provider: str = Field("openai", description="TTS provider: openai, azure, elevenlabs")

    @field_validator('provider')
    def validate_provider(cls, v):
        allowed = {"openai", "azure", "elevenlabs"}
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return v

class TTSResponse(BaseModel):
    audioData: str
```

### tts\app\services\azure_service.py
```py
from pipecat.services.azure import AzureTTSService
from .base import BaseTTSService
import os

class MyAzureTTSService(BaseTTSService):
    def __init__(self):
        azure_api_key = os.getenv("AZURE_API_KEY")
        if not azure_api_key:
            raise RuntimeError("AZURE_API_KEY not set in .env")
        self.service = AzureTTSService(
            voice="", # some azure voice
        )

    async def run_tts(self, text: str):
        async for frame in self.service.run_tts(text):
            yield frame

```

### tts\app\services\base.py
```py
# services/base.py
from typing import AsyncIterator
from pipecat.services.openai import TTSAudioRawFrame, ErrorFrame
# from pipecat.services.azure import TTSAudioRawFrame, ErrorFrame
# from pipecat.services.elevenlabs import TTSAudioRawFrame, ErrorFrame

# Fix Later
class BaseTTSService:
    async def run_tts(self, text: str) -> AsyncIterator[TTSAudioRawFrame | ErrorFrame]:
        raise NotImplementedError("run_tts must be implemented by subclasses")


```

### tts\app\services\elevenlabs_service.py
```py
from pipecat.services.elevenlabs import ElevenLabsTTSService
from .base import BaseTTSService
import os

class MyElevenLabsTTSService(BaseTTSService):
    def __init__(self):
        eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not eleven_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set in .env")
        self.service = ElevenLabsTTSService(
            voice="", # some elevenlabs voice
        )

    async def run_tts(self, text: str):
        async for frame in self.service.run_tts(text):
            yield frame

```

### tts\app\services\openai_service.py
```py
from pipecat.services.openai import OpenAITTSService
from .base import BaseTTSService
import os

class MyOpenAITTSService(BaseTTSService):
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        self.service = OpenAITTSService(
            voice="nova",
            model="tts-1-hd",
            sample_rate=24000
        )

    async def run_tts(self, text: str):
        async for frame in self.service.run_tts(text):
            yield frame

```
 
