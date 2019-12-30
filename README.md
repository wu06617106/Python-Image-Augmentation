# 簡介
This is Image Augmentation Tool  
Written in Python, OpenCV and PyQt

![image](https://github.com/wu06617106/Python-Image-Augmentation/blob/master/UI%20View.jpg)

# 快速入門

1.安裝流程  
```
conda create -n $your-env-name python=3.6.7  
activate $your-env-name
pip install PyQt5, PyQt5-tools, numpy, opencv-python, pydicom, pascal-voc-writer 
```	
Go to PythonImageAugmentation.py file location
```
python PythonImageAugmentation.py
```

2.軟體相依性  
>Python 3.6.7  
>PyQt5 5.11.3  
>pyqt5-tools 5.11.2.1.3  
>numpy 1.15.3  
>opencv-python 3.4.3.18  
>pydicom 1.2.1  
>pascal-voc-writer 0.1.4  
    
3.最新版本  
>Tools Version: 1.5.7   
>Release Date: 2019/12/10  

4.API 參考  
>def Rotate(self, image, angle, center=None, scale=1.0)  
>def Sharpen(self, image):  
>def Contrast(self, image, alpha, beta):  
>def GaussianNoise(self, image, mean, sigma):  
>def Fliplr(self, image, flipCode):  
>def SaltAndPepper(self, image, s_vs_p, amount):  
>def Pad(self, image, offsetX, offsetY):  

5.輸出邏輯  
* **Rotate**  
Range 0 ~ 355  
Step:5, 10, 15, 20, 25, 30  
      
* **Sharpen**  
開啟或關閉   
      
* **Contrast**  
Alpha: 1.0 ~ 3.0  
Beta 0 ~ 100    

* **Gaussian Noise**  
設定三種通道的Mean與Sigma  
Mean B:0 ~ 255  
Mean G:0 ~ 255  
Mean R:0 ~ 255  
Sigma B:0 ~ 255  
Sigma G:0 ~ 255  
Sigma R:0 ~ 255  

* **Fliplr**  
可選擇三種翻轉方式
Vertical, Horizontal, Both  

* **Salt and Pepper**  
調整salt 與 pepper間的比例
S_VS_P: 0 ~ 1.0  
調整noise的數量
Amount: 1 ~ 100  

* **Pad**  
設定哪個方向進行變形  
Top, Bottom, Left, Right  
Top & Left, Top & Right  
Bottom & Left, Bottom & Right  
設定要變形的比例(依照原始影像長寬計算)
Offset: 1 ~ 90  

* **Crop**  
設定ROI區域放大並且裁切  
Ratio: 1 ~ 50  

* 使用者使用此工具時，如果將要套用的多個  
處理方法打勾，就會全部應用至原始圖像。  
例如:選擇套用Fliplr與Sharpen，就會產生  
經過這兩種處理的一張影像。  

__測試部分:__  
>**1.Open File:**  
>執行PythonImageAugmentation.py  
>點選Open選取資料夾  
>程式會讀取該資料夾所有支援的影像  
>使用者能以下拉式選單選擇要開啟的檔案  
>如果開啟DICOM檔案中有一個以上的Frame  
>則可以拖曳Slider來選擇要處理的Frame  
>**2.Image Augmentation**    
>工具右側區塊為各種影像處理  
>使用者能勾選欲套用之處理效果  
>每個效果都有參數能調整  
>勾選完畢後，按下Process按鈕  
>就會對原始影像進行處理  
>**3.ROI**  
>選取ROI來Label影像的Lesion區域  
>框選完畢後按下空白鍵或是Enter來確認選取  
>若要取消ROI，則按下按鍵c  
>確認選取後工具就會對輸入影像進行處理  
>跳出的新視窗顯示處理的結果  
>此時就能按下Save按鈕，選擇儲存位置  
>**4. ROI Save**  
>使用者可以使用ROISave，直接對原始影像擷取ROI並儲存    
>**5. ROI Square**  
>與ROI Save不同之處在於  
>ROI Square功能會自動將使用者框選的ROI之短邊擴增到與長邊等長，  
>並且將邊長增加30%，使得輸出永遠是一個正方形，  
>除此之外還會額外產生四個Crop ROI。
>**6. DICOM to Image**  
>能將DICOM裡面的所有Frame存成BMP圖檔
