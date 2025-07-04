# 🤖🔢 DigitAgent v2

Kaynak klasördeki PNG'leri inceler, Gemma-3 4B IT modeline  
**“What are the numbers?”** sorusunu sorar ve yanıtı  
`{"digits": "123456"}` formatında alarak resmi **digits.png** adıyla hedef klasöre kopyalar.

## Kurulum

```bash
cd /Users/emrealtay/Documents/digit-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
