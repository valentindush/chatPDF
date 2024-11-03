# ChatPDF
Ask questions about multiple pdf files at once. <br/>

Made using `Streamlit`, `LangChain`, `HuggingFaceInferenceAPIEmbeddings`, `Pypdf`,and `FAISS` 

## Installation
#### API Keys
1. **GROQ_API_KEY** To get your Groq API key, go to [https://console.groq.com/keys](https://console.groq.com/keys) and get your own right away.
2. **HF_API_KEY** To get your Hugging Face, go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and get yours.

### Install Packages
To void errors you would want to use **venv**.

```bash
pip install -r requirements.txt
```

### Create .env file
```
HF_API_KEY=<your api key>
GROQ_API_KEY=<your api key>
```

### Run
```bash
streamlit run main.py
```
The application will be accessible on [http://localhost:8501/](http://localhost:8501/)

## Contributing
FEEL FREE !

## LICENSE
Honestly... I don't like them!