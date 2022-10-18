## k近傍法による文字検出
### 操作方法のメモ書き
**VisualStudioにてOpenCVのプロジェクトを２つ作る必要があります**

1. GenData2の方のプロジェクトを実行し，指定した学習用の画像において文字に枠が順次表示されるので枠で囲まれている文字をキーボードで打っていく→全ての文字を打ち終えたらimages.xmlとclassifications.xmlが生成される

2. 生成されたimages.xmlとclassifications.xmlをもう一つのプロジェクト(TrainTest2)にコピー&ペーストする

3. TrainTest2の方のプロジェクトを実行し指定したテスト用画像の文字を検出させると文字認識の結果としてwindow上に認識された文字が出力される

#### 参考文献
Dahms, C (2017) OpenCV_3_KNN_Character_Recognition_Cpp [Source code]. https://github.com/MicrocontrollersAndMore/OpenCV_3_KNN_Character_Recognition_Cpp .
