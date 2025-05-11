

# AICE - AI Model Serving with Flask

이 리포지토리는 Keras 모델을 Flask 서버에 배포하여 실시간으로 예측을 수행하는 시스템입니다. 이 프로젝트는 AI 모델을 웹 서비스로 제공하기 위한 기본적인 서빙 구조를 제공합니다. `Flask`를 사용하여 모델을 API로 서빙하고, `Keras` 모델을 로딩하여 예측을 처리합니다.

## 🚀 **기능**
- **AI 모델 서빙**: 학습된 Keras 모델을 Flask API를 통해 실시간 예측
- **전처리 및 후처리**: 요청에 대해 자동으로 전처리하고 모델의 예측 결과를 후처리
- **Flask 기반**: Flask를 사용하여 서버를 실행하고 API를 제공

## 🛠️ **기술 스택**
- **프로그래밍 언어**: Python
- **프레임워크**: Flask
- **라이브러리**: Keras, NumPy
- **기타**: `aicentro` 라이브러리

## ⚡ **주요 구성 요소**
### 1. **Keras 모델 로더**
Keras로 학습한 `.h5` 모델 파일을 로드하여 예측을 수행하는 로더 클래스입니다.

```python
from aicentro.loader.keras_loader import KerasLoader

loader = KerasLoader(
    model_filename='iris-classification'  # 저장된 모델 파일명 ( .h5 제외 )
)


### 2. **CustomServing 클래스**

Flask에서 사용할 서빙 클래스를 정의하여, 요청에 대한 전처리와 후처리를 자동화합니다.

```python
from aicentro.serving.base_serving import BaseServing
from flask import request, jsonify

class CustomServing(BaseServing):
    def __init__(self, loader, inputs_key='inputs', outputs=None, labels=[]):
        super().__init__(loader=loader, inputs_key=inputs_key, outputs=outputs)
        self.labels = labels

    def pre_processing(self, hash_url):
        _json = request.get_json(silent=True)
        self.inputs = _json[self.inputs_key]

    def post_processing(self, hash_url, response):
        resp_dict = dict()
        resp_dict['classification'] = self.labels[np.argmax(response.reshape(-1))]
        return resp_dict
```

### 3. **Flask 서버 설정**

Flask 서버를 설정하고 모델 서빙을 위한 URL 경로를 지정합니다.

```python
from flask import Flask
from aicentro.serving.serving_config import configure_error_handlers

app = Flask(__name__)

# Flask 객체에 URL Rule 정의
app.add_url_rule(
    '/<hash_url>/',  # URL Path
    view_func=serving,  # 서빙 클래스 객체
    methods=['GET', 'POST']  # HTTP 메소드
)

configure_error_handlers(app=app, code='99', msg_fn=message_format)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

## 📦 **설치 방법**

1. **레포지토리 클론**:

   ```bash
   git clone https://github.com/hyunaeee/AICE.git
   cd AICE
   ```

2. **필요한 라이브러리 설치**:

   ```bash
   pip install -r requirements.txt
   ```

3. **서버 실행**:

   ```bash
   python app.py
   ```

4. **API 접속**:

   * `http://localhost:5000/{hash_url}/`으로 접속하여 모델 예측 서비스를 이용할 수 있습니다.

## 📚 **사용 예시**

```python
import requests
import json

url = 'http://localhost:5000/your_hash_url/'

data = {
    "inputs": [5.1, 3.5, 1.4, 0.2]  # 예시: Iris 데이터 샘플
}

response = requests.post(url, json=data)
print(response.json())
```

## 🧑‍🤝‍🧑 **기여 방법**

1. 이 레포지토리를 Fork합니다.
2. 자신의 브랜치에서 작업을 합니다.
3. Pull Request를 통해 기여합니다.


## 📝 **TODO List**

* [ ] 다국어 지원 추가
* [ ] 클라우드 배포 설정


