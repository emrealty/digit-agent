# 🤖🔢 DigitAgent

**DigitAgent**, klasörünüzdeki PNG biçimindeki sayı-CAPTCHA (veya benzeri) görselleri  
Google **Gemma 3 4B IT** modeline çözdürerek dosya adlarını
`<bulunan_rakamlar>_<orijinal_dosya>.png` biçiminde günceller.

---

## Kurulum

```bash
git clone https://github.com/<kullanici>/digit-agent.git
cd digit-agent
python -m venv venv && source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
