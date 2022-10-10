// 学習に使う画像とカテゴリーの対応情報と画像データをxmlファイルとして生成

#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h> // for CV_BGR2GRAY constant


using namespace cv;
using namespace std;

const int MIN_CONTOUR_AREA = 70; // 細めのフォントでも確実に検出されるように
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;
// 統一した大きさの四角内に文字を収めるので


int main() {
	Mat inputTrainingImg; // 読み込むtraining用画像
	Mat grayScaledImg;
	Mat thresholdedImg;
	Mat clonedThresholdedImg;

	// 輪郭検出用
	vector<vector<Point>> contoursVector; // 検出される輪郭の座標
	vector<Vec4i> vecHiearchy; // 階層構造の情報を保存した配列

	Mat classificationForTraining; // 学習のための画像とカテゴリーの対応情報
	Mat imagesForTrainingAsFlattenedFloat;

	// 学習させるカテゴリーの限定
	vector<int> categoriesOfInterest = { '1','7','2','J','Z','S' }; // waitKeyの返り値の型に合わせるためint型

	inputTrainingImg = imread("training3.png");

	// グレースケール化(二値化のために)
	cvtColor(inputTrainingImg, grayScaledImg, CV_BGR2GRAY);

	// 二値化処理(より高精度な輪郭抽出のため)
	adaptiveThreshold(grayScaledImg, thresholdedImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
	imshow("thresholded", thresholdedImg);

	clonedThresholdedImg = thresholdedImg.clone(); // findContoursの際に上書きされてしまうため複製しておく

	// 輪郭検出(一番外側の輪郭のみ,階層構造なし)
	findContours(clonedThresholdedImg, contoursVector, vecHiearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contoursVector.size(); i++) {
		if (contourArea(contoursVector[i]) > MIN_CONTOUR_AREA) { // 小さすぎるものは無視
			Rect rect = boundingRect(contoursVector[i]); 	// 検出した輪郭ベクトルから成る点群に外接する最小の長方形
			rectangle(inputTrainingImg, rect, Scalar(0, 0, 255), 2); // 原画像に囲んでいる四角形を描画

			Mat roi = thresholdedImg(rect); // 囲まれたregionのみinterested

			Mat resizedRoi;
			resize(roi, resizedRoi, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)); // region of interestをすべて同じ面積にそろえる

			imshow("roi", roi);
			imshow("resized roi", resizedRoi);
			imshow("原画像inputTrainingImg", inputTrainingImg); // キーボード入力する際に現在のroiが分かるようimshowする

			int inputIntChar = waitKey(0); // キーボード入力した文字/数字をintとして返す

			if (find(categoriesOfInterest.begin(), categoriesOfInterest.end(), inputIntChar) != categoriesOfInterest.end()) {
				// もしキーボード入力したものがcategoriesOfInterestに含まれているものなら
				classificationForTraining.push_back(inputIntChar);

				Mat imgAsFloat;
				resizedRoi.convertTo(imgAsFloat, CV_32FC1); // resizeしておいたroiをMat型からfloat型に変換

				Mat imgAsFlattenedFloat = imgAsFloat.reshape(1, 1);

				imagesForTrainingAsFlattenedFloat.push_back(imgAsFlattenedFloat);
			}
		}
	}

	// classificationForTrainingをclassification.xmlとして書き出す
	FileStorage fileStorageClassifications("classifications.xml", FileStorage::WRITE);
	fileStorageClassifications << "classfications" << classificationForTraining; 
	fileStorageClassifications.release();

	// imagesForTrainingAsFlattenedFloatをimages.xmlとして書き出す
	FileStorage fileStorageImages("images.xml", FileStorage::WRITE);
	fileStorageImages << "images" << imagesForTrainingAsFlattenedFloat;
	fileStorageImages.release();

	return 0;
}
