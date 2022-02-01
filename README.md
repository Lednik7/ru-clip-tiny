# RuCLIPtiny
Zero-shot image classification model for Russian language

---

RuCLIPtiny (Russian Contrastive Language–Image Pretraining) is a neural network trained to work with different pairs (images, texts). Our model is based on [ConvNeXt-tiny](https://github.com/facebookresearch/ConvNeXt) and [DistilRuBert-tiny](https://huggingface.co/DeepPavlov/distilrubert-tiny-cased-conversational-v1), and is supported by extensive research zero-shot transfer, computer vision, natural language processing, and multimodal learning.

# Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-l2JtynS3ZwlE8g5wNYNdTYUVQRLWl9m?usp=sharing)
Evaluate & Simple usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lojdsARMzsURUkKJQLlEQAPyvM99U12n?usp=sharing)
Finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yl1oVem3Pw0o1ZlukR2Fg9dGyqRFeu1z?usp=sharing)
ONNX conversion and speed testing

# Usage
Install rucliptiny module and requirements first. Use this trick
```python3
!gdown -O ru-clip-tiny.pkl https://drive.google.com/uc?id=1-3g3J90pZmHo9jbBzsEmr7ei5zm3VXOL
!pip install git+https://github.com/cene555/ru-clip-tiny.git
```
## Example in 3 steps
0. Download CLIP image from repo
```python3
!wget -c -O CLIP.png https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
```
0. Import libraries
```python3
import torch
from torchvision import transforms
import transformers
from rucliptiny import RuCLIPtiny
from rucliptiny.predictor import Predictor
from PIL import Image
from rucliptiny.utils import get_transform
from rucliptiny.tokenizer import Tokenizer

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
2. Load model, transforms, tokenizer
```python3
model = RuCLIPtiny()
model.load_state_dict(torch.load('/content/ru-clip-tiny.pkl', map_location=device))
model = model.to(device).eval()
for x in model.parameters(): x.requires_grad = False
torch.cuda.empty_cache()

transforms, tokenizer = get_transform(), Tokenizer()
```
3. Preprocess image and texts
```python3
# batch first
image = transforms(Image.open("CLIP.png")).unsqueeze(0).to(device) # [1, 3, 224, 224]

# batch first
texts = ['диаграмма', 'собака', 'кошка']
input_ids, attention_mask = tokenizer.tokenize(texts, max_len=77)
input_ids, attention_mask = text_tokens.to(device), attention_mask.to(device) # [3, 77]
```
4. Simple inference
```python3
image_features = self.encode_image(image)
text_features = self.encode_text(input_ids, attention_mask)

logits_per_image, logits_per_text = model(image, input_ids, attention_mask)
probs = logits_per_image.softmax(dim=-1).cpu().numpy()
```
