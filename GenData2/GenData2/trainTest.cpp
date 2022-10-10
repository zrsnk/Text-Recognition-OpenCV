 #include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
 #include<opencv2/ml/ml.hpp>
 #include<iostream>
#include<sstream>
#include <opencv2/imgproc/types_c.h> // for CV_BGR2GRAY constant

using namespace cv;
using namespace std;

const int MIN_CONTOUR_AREA = 70;
const int RESIZED_IMAGE_SIZE = 30;

class ContourWithData {
public:
	vector<Point> contoursVector; // 検出した輪郭ベクトル
	Rect rect; // 一番外側の輪郭に外接する四角形格
	float contourArea; // 検出した輪郭の面積

	bool isContourConsiderable() { // 小さすぎる面積の輪郭はfalseを返してはじく
		return !(contourArea < MIN_CONTOUR_AREA);
	}

	// 囲んでいる四角を左からソート
	static bool sortByRectCPosition(const ContourWithData& contour1, const ContourWithData& contour2) {
		return (contour1.rect.x < contour2.rect.x); // x座標を比較して小さい方(左にある方)を返していく 
	}
};

int main() {
	// 1. サンプル画像データとカテゴリーを対応付けるよう学習

	vector<ContourWithData> contoursWithData;
	vector<ContourWithData> filteredContoursWithData;

	Mat classificationForTraining; // classifications.xmlから読み込んだデータ
	Mat imagesForTrainingAsFlattenedFloat; // images.xmlから読み込んだデータ

	// 学習用データを読み込み
	FileStorage classifications("classifications.xml", FileStorage::READ);
	classifications["classifications"] >> classificationForTraining;
	classifications.release();

	FileStorage images("images.xml", FileStorage::READ);
	images["images"] >> imagesForTrainingAsFlattenedFloat;
	images.release();

	Ptr<ml::KNearest> kNearest(ml::KNearest::create()); // kNearestオブジェクトをインスタンス化

	// 集めた学習用データで学習
	kNearest->train(imagesForTrainingAsFlattenedFloat, ml::ROW_SAMPLE, classificationForTraining);


	// 2. 学習した(識別辞書が準備できた)ので与えられたテスト画像を識別処理
	Mat inputTestingImg = imread("test1.png");

	// 前処理
	Mat grayScaledImg;
	Mat thresholdedImg;
	Mat clonedThresholdedImg;

	cvtColor(inputTestingImg, grayScaledImg, CV_BGR2GRAY);


	return 0;
}
