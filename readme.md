# 🧳 여행 짐 싸기 도우미, 패키 (Packy)

본 프로젝트는 여행 준비 과정에서 발생하는 짐 싸기의 번거로움과 정보 검색의 비효율성을 해결하기 위해 기획되었습니다.
단순히 일반적인 준비물을 나열하는 것을 넘어, RAG 기술을 활용해 항공 규정과 현지 상황을 반영한 '맞춤형 짐 싸기 가이드'를 제공하는 것을 목표로 합니다.

## 1. 프로젝트 소개
여행지마다 전압, 기후, 문화적 특성이 달라 필요한 준비물이 제각각입니다. 기존의 정보 검색 방식은 여러 사이트를 직접 찾아봐야 하는 번거로움이 있었습니다.
이를 해결하기 위해 사용자의 목적지에 맞춰 필수품부터 찐 고수들의 꿀팁까지 자동으로 분석하고 제안하는 AI 에이전트 '패키(Packy)'를 개발했습니다.

## 2. 사용 기술 (Tech Stack)

언어 및 모델
- Python 3.12
- OpenAI GPT-4o

프레임워크 및 라이브러리
- LangChain (Core, Community)
- Streamlit (웹 인터페이스 구현)
- ChromaDB (벡터 데이터베이스)

도구 (Tools)
- DuckDuckGo Search (실시간 웹 검색)
- PyPDFLoader (PDF 문서 처리)

## 3. 프로젝트 구조
유지보수가 용이하도록 기능별로 파일을 분리한 클래스 기반 모듈형 구조입니다.

📦 travel-packing-helper
├── data                  # RAG 지식 데이터 (기본템, 꿀팁, 규정 등)
├── modules               # 핵심 기능 모듈
│   ├── agent.py          # ReAct 에이전트 (사고 과정 및 행동 정의)
│   ├── llm.py            # 모델 설정 (GPT-4o)
│   ├── vector_store.py   # 데이터 임베딩 및 검색 관리
│   ├── tools.py          # 검색 도구 연결
│   ├── history.py        # 대화 내용 기억 (Memory)
│   └── logger.py         # 시스템 로그 기록
├── main.py               # 메인 실행 파일 (Streamlit UI)
├── requirements.txt      # 의존성 패키지 목록
└── app.log               # 실행 로그 파일

## 4. 주요 기능

(1) 체계적인 3단계 답변 시스템
사용자의 혼란을 줄이기 위해 답변을 3가지 섹션으로 구조화하여 제공합니다.
1. 🎒 필수템: 여권, 전압(돼지코), 상비약 등 놓치면 안 되는 기본 물품
2. ✨ 센스 꿀팁: 경험자만 아는 생활 밀착형 꿀팁 (장거리 비행 팁 등)
3. 🌏 현지 맞춤 아이템: 여행지 특성에 맞는 특별 아이템 (웹 검색 활용)

(2) 도메인 특화 데이터 활용 (RAG)
일반적인 LLM이 놓칠 수 있는 디테일한 정보를 제공합니다.
- 항공 규정: 무선 고데기, 보조배터리 등 헷갈리는 기내 반입 규정 안내
- 현지 정보: 일본 동전 지갑, 동남아 샤워기 필터 등 국가별 필수템 추천

(3) 실시간 정보 보강 (Web Search)
내부 데이터에 없는 정보나 최신 유행 아이템이 필요한 경우, DuckDuckGo Search Tool을 통해 실시간으로 정보를 검색하여 답변을 보강합니다.

(4) 시스템 로그 기록
운영 관점을 고려하여 사용자의 질문과 AI의 답변, 시스템 에러 로그를 app.log 파일에 자동으로 기록합니다.

## 5. 실행 방법

1. 저장소 다운로드
git clone https://github.com/jjoo22/1202-project-travel-packing-helper-.git
cd 1202-project-travel-packing-helper-

2. 패키지 설치
pip install -r requirements.txt

3. API 키 설정
프로젝트 폴더에 .env 파일을 생성하고 아래 내용을 입력합니다.
OPENAI_API_KEY=sk-본인의_키_입력

4. 프로그램 실행
streamlit run main.py