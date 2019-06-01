# 簡介
This is Image Augmentation Tool  
Written in Python, OpenCV and PyQt

![image](https://github.com/wu06617106/Python-Image-Augmentation/blob/master/UI%20View.jpg)

# 快速入門

1.安裝流程  
```
conda create -n $your-env-name python=3.6.7  
activate $your-env-name
pip install PyQt5 opencv-python pydicom pascal-voc-writer
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
>Tools Version: 1.5.6  
>Release Date: 2019/04/01  

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
**最多有71種角度輸出**  
      
* **Sharpen**  
開啟或關閉  
**有2種輸出**  
      
* **Contrast**  
Alpha: 1.0 ~ 3.0  
Beta 0 ~ 100  
**有2121種輸出**  

* **Gaussian Noise**  
設定三種通道的Mean與Sigma  
Mean B:0 ~ 255  
Mean G:0 ~ 255  
Mean R:0 ~ 255  
Sigma B:0 ~ 255  
Sigma G:0 ~ 255  
Sigma R:0 ~ 255  
**256^6種輸出**  

* **Fliplr**  
可選擇三種翻轉方式
Vertical, Horizontal, Both  
**3種輸出**  

* **Salt and Pepper**  
調整salt 與 pepper間的比例
S_VS_P: 0 ~ 1.0  
調整noise的數量
Amount: 1 ~ 100  
**1100種輸出**  

* **Pad**  
設定哪個方向進行變形  
Top, Bottom, Left, Right  
Top & Left, Top & Right  
Bottom & Left, Bottom & Right  
設定要變形的比例(依照原始影像長寬計算)
Offset: 1 ~ 90  
**720種輸出**  

* **Crop**  
設定ROI區域放大並且裁切  
Ratio: 1 ~ 50  
**50種輸出**  

* 使用者使用此工具時，如果將要套用的多個  
處理方法打勾，就會全部應用至原始圖像。  
例如:選擇套用Fliplr與Sharpen，就會產生  
經過這兩種處理的一張影像。  

# 建置與測試  
__建置部分:__  
>本專案以Visual Studio 2017開發  
>Visual Studio 2017必須安裝Python開發環境  

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

# 貢獻
>1.0.0版:  
    支援JPG以及PNG檔案，影像處理效果包含Rotate、Sharpen、Contrast、  
    GaussianNoise、Fliplr、SaltAndPepper、Pad  
>1.2.0版:  
    Rotate方法改變為，輸入一個範圍與設定一個Step，  
    程式會在選定範圍內，依照Step來逐一產生Rotate之影像。(Step：每次增  
    加幾度)   
>1.3.0版：  
    支援DICOM檔案  
>1.3.1版:  
    修正Salt and Pepper method  
    優化Contrast method.  
>1.3.2版:  
    修正Affine Transform method  
    修正一些參數的錯誤  
>1.3.3版:  
    增加pascal_voc_writer  
    輸出影像會額外產生相對應的PASCAL VOC檔案  
>1.4.0版:  
    增加Crop method  
>1.4.1版:  
    支援BMP檔案  
>1.5.0版:  
	支援ROI存檔  
	開檔模式改成選取資料夾，程式會讀取所有影像名稱，  
	使用者可以用下拉式選單選擇檔案。  
>1.5.1版:  
	修正讀取DCM檔案，OpenCV處理時沒有轉成BGR的問題。  
>1.5.2版:  
	新增ROI Square功能，自動將選取的ROI短邊擴增到與長邊相等，輸出永遠為正方形。  
>1.5.3版:  
	更新ROI Square功能，輸出更改為原本正方形ROI的1.3倍。  
>1.5.4版:  
	更新ROI Square功能，額外輸出四個Crop ROI影像。  
>1.5.5版:  
	新增DICOM to Image功能，將DICOM裡面的所有Frame存成BMP圖檔。  
	修正讀取DICOM檔案時，Number of Frames讀取異常的問題。  
>1.5.6版:  
	修正DICOM中NumberOfFrames這個Tag如果是空的問題。  
