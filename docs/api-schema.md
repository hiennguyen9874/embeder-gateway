Dưới đây là **API outbound**: format WeKnora gửi tới embedding/reranking provider, **không phải API client gọi vào WeKnora**. Phạm vi là snapshot **v0.8.3, revision `fd7a2586fd58`**.

WeKnora chọn protocol theo provider, model catalog và các override. Vì vậy cùng một provider có thể dùng format khác nhau cho text và multimodal. Các JSON dưới đây là ví dụ tối giản theo code; trường optional, URL và auth có thể thay đổi theo cấu hình.

## A. Embedding

### 1. OpenAI-compatible — text

Dùng cho OpenAI và nhiều provider/gateway tương thích.

```http
POST {base_url}/embeddings
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

**Request**

```json
{
  "model": "embedding-model",
  "input": ["Văn bản A", "Văn bản B"]
}
```

**Response**

```json
{
  "data": [
    {"index": 0, "embedding": [0.12, -0.04, 0.31]},
    {"index": 1, "embedding": [0.08, 0.15, -0.22]}
  ]
}
```

- `index` ánh xạ vector về input. Client cũng chấp nhận response không có `index` và đọc theo thứ tự.
- Các trường như `dimensions`, `encoding_format`, `input_type`, `task` hoặc `truncate` phụ thuộc model/provider; không phải lúc nào WeKnora cũng gửi.
- Azure OpenAI có thể dùng endpoint/auth riêng, không mặc định áp dụng Bearer như ví dụ.

Nguồn: [OpenAI embedding client](raw/WeKnora/internal/models/api/openaiembeddings/client.go), [provider authentication](raw/WeKnora/internal/models/providers/definition.go).

### 2. OpenAI-compatible — image extensions

Không có một image format thống nhất cho mọi server tương thích OpenAI. WeKnora hỗ trợ các biến thể được cấu hình:

**Image object**, ví dụ Jina:

```json
{
  "model": "jina-embeddings-v5-omni-small",
  "input": [
    {"image": "data:image/png;base64,<BASE64_IMAGE>"}
  ],
  "truncate": true
}
```

**Messages format**, ví dụ một số model qua vLLM-compatible server:

```json
{
  "model": "TIGER-Lab/VLM2Vec-Full",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}
        }
      ]
    }
  ]
}
```

Response vẫn được client đọc theo dạng `data[].embedding`. Model **và** endpoint phải được khai báo hỗ trợ ảnh; chỉ trỏ một OpenAI-compatible URL tới model vision là chưa đủ. [Image request builder](raw/WeKnora/internal/models/api/openaiembeddings/client.go), [các ví dụ wire trong test](raw/WeKnora/internal/models/embedding/image_wire_test.go).

### 3. Aliyun DashScope native — multimodal

Text embedding thông thường của Aliyun có thể dùng OpenAI-compatible; các model multimodal dùng adapter native:

```http
POST {host}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

**Text request**

```json
{
  "model": "qwen3-vl-embedding",
  "input": {
    "contents": [{"text": "Biểu đồ doanh thu"}]
  }
}
```

**Image request**

```json
{
  "model": "qwen3-vl-embedding",
  "input": {
    "contents": [{"image": "data:image/png;base64,<BASE64_IMAGE>"}]
  }
}
```

Khi model và cấu hình cho phép đổi dimension, request có thể thêm:

```json
"parameters": {"dimension": 1024}
```

**Response**

```json
{
  "output": {
    "embeddings": [
      {"index": 0, "embedding": [0.12, 0.34], "type": "image"}
    ]
  }
}
```

Nguồn: [DashScope embedding client](raw/WeKnora/internal/models/api/dashscopeembeddings/client.go), [Aliyun endpoint](raw/WeKnora/internal/models/providers/aliyun.go).

### 4. Volcengine Ark native — multimodal

```http
POST {host}/api/v3/embeddings/multimodal
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

**Text**

```json
{
  "model": "doubao-embedding-vision-251215",
  "input": [{"type": "text", "text": "Biểu đồ doanh thu"}]
}
```

**Image**

```json
{
  "model": "doubao-embedding-vision-251215",
  "input": [
    {
      "type": "image_url",
      "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}
    }
  ],
  "encoding_format": "float"
}
```

**Response**

```json
{
  "data": {
    "object": "embedding",
    "embedding": [0.12, 0.34]
  }
}
```

Ark trả **một vector cho toàn bộ input**. Adapter WeKnora vì thế gửi một text hoặc một image mỗi request, không coi mảng `input` là batch vector độc lập. Model text cũ có thể được route sang OpenAI-compatible thay vì format này. [Ark embedding client](raw/WeKnora/internal/models/api/arkembeddings/client.go), [wire examples](raw/WeKnora/internal/models/embedding/image_wire_test.go).

### 5. Google Gemini native

```http
POST {base_url}/models/{model}:batchEmbedContents
x-goog-api-key: <API_KEY>
Content-Type: application/json
```

**Request text và image là các request con độc lập:**

```json
{
  "requests": [
    {
      "model": "models/gemini-embedding-2",
      "content": {"parts": [{"text": "Biểu đồ doanh thu"}]}
    },
    {
      "model": "models/gemini-embedding-2",
      "content": {
        "parts": [
          {
            "inlineData": {
              "mimeType": "image/png",
              "data": "<BASE64_IMAGE>"
            }
          }
        ]
      }
    }
  ]
}
```

**Response**

```json
{
  "embeddings": [
    {"values": [0.12, 0.34]},
    {"values": [0.56, 0.78]}
  ]
}
```

Vectors được ghép theo **thứ tự** request con. Khi được cấu hình, options như `outputDimensionality` nằm trong `embedContentConfig` của từng request. WeKnora thực hiện text và image qua các lời gọi riêng; ví dụ trên minh họa shape của từng request con, không khẳng định adapter trộn hai modality trong một batch thực tế. [Gemini client](raw/WeKnora/internal/models/api/googleembeddings/client.go), [provider config](raw/WeKnora/internal/models/providers/gemini.go).

### 6. Ollama local

WeKnora gọi Ollama Go SDK với `EmbedRequest`. Body logic tương ứng:

```json
{
  "model": "nomic-embed-text",
  "input": ["Văn bản A", "Văn bản B"],
  "truncate": true,
  "options": {"num_ctx": 511}
}
```

Kết quả được đọc từ `embeddings`, mỗi input một vector. `dimensions` chỉ được đặt nếu model/config cho phép. Trong phần source đã kiểm tra, lời gọi đi qua SDK `client.Embed`; tôi chưa xác minh HTTP path do SDK tạo ra, nên không khẳng định path ở đây. [Ollama adapter](raw/WeKnora/internal/models/embedding/ollama.go), [SDK call](raw/WeKnora/internal/models/utils/ollama/ollama.go).

---

## B. Reranking

### 1. Cohere-compatible — text

Đây là format chung cho Cohere và nhiều gateway/provider như Jina, Zhipu, SiliconFlow, Qianfan, GPUStack và Generic.

```http
POST {base_url}/rerank
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

**Request**

```json
{
  "model": "rerank-model",
  "query": "Doanh thu năm nay?",
  "documents": [
    "Doanh thu tăng 20%.",
    "Thông tin nhân sự."
  ]
}
```

**Response**

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 1, "relevance_score": 0.12}
  ]
}
```

Client cũng chấp nhận `score` thay `relevance_score`, hoặc mảng kết quả trực tiếp. `top_n`, `return_documents` và các optional fields chỉ được gửi theo cấu hình provider/model. [Cohere-style client](raw/WeKnora/internal/models/api/cohererank/client.go).

### 2. Cohere-style — image, ví dụ Jina

```json
{
  "model": "jina-reranker-m0",
  "query": "Biểu đồ nào thể hiện doanh thu?",
  "documents": [
    {"image": "data:image/png;base64,<BASE64_IMAGE>"}
  ],
  "return_documents": false
}
```

Response vẫn dùng `results`/`index`/score như trên. **Giới hạn quan trọng:** comment trong code nói Jina image input dạng data URI ở rerank **chưa được xác minh với API thật**; test source dùng mock upstream. [Client](raw/WeKnora/internal/models/api/cohererank/client.go), [image wire test](raw/WeKnora/internal/models/rerank/image_wire_test.go).

### 3. Aliyun DashScope native — text

```http
POST {host}/api/v1/services/rerank/text-rerank/text-rerank
Authorization: Bearer <API_KEY>
```

**Request**

```json
{
  "model": "gte-rerank-v2",
  "input": {
    "query": "Doanh thu năm nay?",
    "documents": ["Doanh thu tăng 20%.", "Thông tin nhân sự."]
  },
  "parameters": {
    "return_documents": true,
    "top_n": 2
  }
}
```

**Response**

```json
{
  "output": {
    "results": [
      {
        "index": 0,
        "relevance_score": 0.95,
        "document": {"text": "Doanh thu tăng 20%."}
      }
    ]
  }
}
```

Adapter này là **text rerank**: `documents` là mảng string. [DashScope rerank client](raw/WeKnora/internal/models/api/dashscoperank/client.go).

### 4. NVIDIA NIM — text và image

**Request text**

```json
{
  "model": "nvidia/nv-rerankqa-mistral-4b-v3",
  "query": {"text": "Doanh thu năm nay?"},
  "passages": [
    {"text": "Doanh thu tăng 20%."},
    {"text": "Thông tin nhân sự."}
  ],
  "truncate": "END"
}
```

**Request image**, với model vision hỗ trợ:

```json
{
  "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
  "query": {"text": "Biểu đồ nào thể hiện doanh thu?"},
  "passages": [
    {"image": "data:image/png;base64,<BASE64_IMAGE>"}
  ],
  "truncate": "END"
}
```

**Response**

```json
{
  "rankings": [
    {"index": 0, "logit": 2.1}
  ]
}
```

`logit` không phải xác suất 0–1; WeKnora áp dụng sigmoid khi score scale được cấu hình là logit. [NIM client](raw/WeKnora/internal/models/api/nimrerank/client.go), [score normalization](raw/WeKnora/internal/models/rerank/protocol.go).

### 5. Pinecone native

```http
POST https://api.pinecone.io/rerank
api-key: <API_KEY>
X-Pinecone-Api-Version: 2026-07
Content-Type: application/json
```

**Request**

```json
{
  "model": "bge-reranker-v2-m3",
  "query": "Doanh thu năm nay?",
  "documents": [
    {"text": "Doanh thu tăng 20%."},
    {"text": "Thông tin nhân sự."}
  ],
  "top_n": 2,
  "rank_fields": ["text"],
  "return_documents": false
}
```

**Response**

```json
{
  "data": [
    {"index": 0, "score": 0.95},
    {"index": 1, "score": 0.12}
  ]
}
```

WeKnora yêu cầu đủ một score cho mỗi document và kiểm tra index trùng hoặc ngoài phạm vi. Header version trên là giá trị trong snapshot này. [Pinecone client](raw/WeKnora/internal/models/api/pineconerank/client.go), [provider defaults](raw/WeKnora/internal/models/providers/pinecone.go).

### 6. Hugging Face TEI native

```http
POST {base_url}/rerank
Content-Type: application/json
```

**Request**

```json
{
  "query": "Doanh thu năm nay?",
  "texts": [
    "Doanh thu tăng 20%.",
    "Thông tin nhân sự."
  ],
  "raw_scores": false,
  "truncate": true
}
```

**Response — mảng trực tiếp**

```json
[
  {"index": 0, "score": 0.95},
  {"index": 1, "score": 0.12}
]
```

Không có `model` trong body: model được chọn lúc triển khai TEI. Auth phụ thuộc server; test của WeKnora có ví dụ Bearer token. [TEI client](raw/WeKnora/internal/models/api/teirank/client.go), [wire test](raw/WeKnora/internal/models/rerank/wire_test.go).

### 7. Tencent LKEAP — `RunRerank`

WeKnora dùng Tencent Cloud SDK và **TC3 signing**, với `SecretId`/`SecretKey`, không phải Cohere-compatible Bearer API.

**Body của action:**

```json
{
  "Query": "Doanh thu năm nay?",
  "Docs": ["Doanh thu tăng 20%.", "Thông tin nhân sự."],
  "Model": "lke-reranker-base"
}
```

**Response**

```json
{
  "Response": {
    "ScoreList": [0.95, 0.12],
    "RequestId": "request-id"
  }
}
```

`ScoreList` tương ứng **theo thứ tự input**, không phải danh sách đã sort. Endpoint mặc định là `lkeap.tencentcloudapi.com`, region mặc định `ap-guangzhou`; SDK xử lý chữ ký và action headers. [LKEAP client](raw/WeKnora/internal/models/api/tencentlkeap/client.go), [wire test](raw/WeKnora/internal/models/rerank/wire_test.go).

### 8. Volcengine Knowledge Service — rerank

Đây là API khác với Ark embedding. WeKnora dùng SDK với **IAM Access Key/Secret Key signing**.

```http
POST {base_url}/api/knowledge/service/rerank
```

**Request**

```json
{
  "rerank_model": "doubao-seed-rerank",
  "rerank_instruction": "Whether the document answers the query or matches the content retrieval intent",
  "datas": [
    {
      "query": "Doanh thu năm nay?",
      "content": "Doanh thu tăng 20%."
    },
    {
      "query": "Doanh thu năm nay?",
      "content": "Thông tin nhân sự."
    }
  ]
}
```

**Response**

```json
{
  "code": 0,
  "message": "success",
  "data": {"scores": [0.95, 0.12]}
}
```

Điểm trong `scores` tương ứng theo thứ tự input. Region mặc định trong client là `cn-beijing`. [Volcengine client](raw/WeKnora/internal/models/api/volcengineknowledge/client.go), [wire test](raw/WeKnora/internal/models/rerank/wire_test.go).

## Tóm tắt để triển khai service tương thích

| Nhu cầu | Format nên dùng |
|---|---|
| Text embedding self-hosted | OpenAI-compatible `model` + `input` → `data[].embedding` |
| Text reranking self-hosted | Cohere-compatible `model` + `query` + `documents` → `results[].index/relevance_score` |
| Image embedding | Chọn đúng biến thể mà model/provider khai báo: image object, messages, DashScope, Ark hoặc Gemini |
| Image reranking | Chọn image-capable model và format tương ứng, ví dụ Jina-style hoặc NVIDIA NIM |

**Giới hạn kiểm chứng:** Các format được tổng hợp từ static inspection của adapter và test source tại revision trên; tôi **không chạy test hay gọi provider thực tế**. Ví dụ response được rút gọn và không bao gồm mọi metadata/error field. Chưa cập nhật tổng hợp này vào wiki.