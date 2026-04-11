使用llama.cpp將HuggingFace 取得的LLM模型轉為 GGUF格式

近期在測試一些模型以及落地的應用，由於本地執行環境我是借用Ollama來實現， Ollama 具有易部署且容器化的便利性，只需要一行就能完成 LLM 的部署，再搭配Open-WebUI直接馬上享有開箱即用的操作及測試，但並非所有的模型都有提供GGUF的格式，因此就得自行轉檔。本篇就記錄一下如何將 HuggingFace 取得的模型轉為 GGUF格式，以便能在Ollama上進行部署使用。

整個作業大致包含以下項目：

註冊HuggingFace帳號，做為模型檔案取得的來源。(AI Model界的github)
Git CLI，模型檔案多數位於github上
llama.cpp，llama.cpp是源自於GGML基於C/C++ 實現，可以用CPU運行模型，除了模型運作之外，也支援做為轉GGUF檔工具，並且也可以進行開源模型的量化處理，不過縮小模型，當然模型能力也會有所下降，像是4 bit 量化後的模型其實滿慘的
Ollama用於部署並執行模型於本機端，支援容器化，簡單暴力好用
Open-WebUI 用於操作及測試模型，相似於OpenAI ChatGPT Web 介面
安裝 llama.cpp
首先clone llama.cpp 到本機
git clone <https://github.com/ggerganov/llama.cpp.git>
2. 由於轉檔程式是使用Python所撰寫，因此需要安裝必要的相依項目

cd llama.cpp
pip install -r requirements.txt
3. 相依套件安裝後，建議檢查一下是否有順利安裝成功

python convert.py -h
下載 HuggingFace 的開源模型
這裡我以taide的taide/Llama3-TAIDE-LX-8B-Chat-Alpha1模型做為示範

進入HuggingFace找到taide的taide/Llama3-TAIDE-LX-8B-Chat-Alpha1模型，切換至Files and versions 頁籤，選擇Clone repository，使用 git clone下載，由於模型檔很大，請確保在網路順暢的環境下載
git clone <https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1>
2. 下載時會要求輸入身份驗證，這裡要注意的是目前已不再支撐使用HuggingFace密碼驗證，因此必須在HuggingFace建立個人的access token，進入個人設定頁，找到Access Tokens產生一組出來用，必須自已Copy下來，否則忘了就只能再建一個新的了。

1. 順利下載後，就可以開始做轉檔作業，進入llama.cpp目錄，執行以下指令，其中../Llama3-TAIDE-LX-8B-Chat-Alpha1指定到你從HuggingFace下載回來的模型目錄，而 outfile則是轉成GGUF的模型檔名

python .\convert_hf_to_gguf.py ../Llama3-TAIDE-LX-8B-Chat-Alpha1 --outfile models/Llama3-TAIDE-LX-8B-Chat-Alpha1.gguf
4.等待轉完後，會有以下訊息

Ollama 載入本地GGUF模型

1. 撰寫Modelfile，其實就是一個文字檔案，FROM後面指向你的GGUF檔實際路徑
FROM Llama3-TAIDE-LX-8B-Chat-Alpha1.gguf

2. 載入模型，create 給與一個名稱(通常給模型名稱)，而 -f 則是指向Modelfile

ollama create Llama3-TAIDE-LX-8B-Chat-Alpha1 -f Llama3-TAIDE-LX-8B-Chat-Alpha1
3. 看到以下文字就是載入成功了

1. 開啟瀏覽器，進入OpenAI-WebUI系統介面後，有看到的部署上去的模型，就沒錯了
