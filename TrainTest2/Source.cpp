#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h> // for CV_BGR2GRAY constant

using namespace cv;
using namespace std;

const int MIN_CONTOUR_AREA = 70;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

class ContourWithData {
public:
	vector<Point> contourVector; // 検出した輪郭ベクトル
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

	// グレースケール化
	cvtColor(inputTestingImg, grayScaledImg, CV_BGR2GRAY);

	// 二値化
	adaptiveThreshold(grayScaledImg, thresholdedImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

	clonedThresholdedImg = thresholdedImg.clone();

	// 輪郭検出用
	vector<vector<Point>> contoursVector; // 検出される輪郭の座標
	vector<Vec4i> vecHiearchy; // 階層構造の情報を保存した配列

	// 輪郭検出(一番外側の輪郭のみ,階層構造なし)
	findContours(clonedThresholdedImg, contoursVector, vecHiearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contoursVector.size(); i++) {
		ContourWithData contourWithData; // contourWithDataオブジェクト インスタンス化

		// それぞれの輪郭ベクトルに対してプロパティに情報を入れていく
		contourWithData.contourVector = contoursVector[i];
		contourWithData.rect = boundingRect(contourWithData.contourVector);
		contourWithData.contourArea = contourArea(contourWithData.contourVector);

		// できあがったそれぞれのオブジェクトを配列に入れていく
		contoursWithData.push_back(contourWithData);


	}


	// contourWithDataの中で面積が十分大きいもののみを集める
	for (int i = 0; i < contoursWithData.size(); i++) {
		if (contoursWithData[i].isContourConsiderable()) {
			filteredContoursWithData.push_back(contoursWithData[i]);
		}
	}

	// テスト画像の文字を必ず左から右の順に認識処理するようにソートしておく
	sort(filteredContoursWithData.begin(), filteredContoursWithData.end(), ContourWithData::sortByRectCPosition);

	string recognizedString;

	// それぞれのrectで囲まれた文字に対して
	for (int i = 0; i < filteredContoursWithData.size(); i++) {
		// 四角い枠線を描画する
		rectangle(inputTestingImg, filteredContoursWithData[i].rect, Scalar(0, 255, 0), 2);

		// thresholdedImgの中でも囲まれた領域のみ注目
		Mat roi = thresholdedImg(filteredContoursWithData[i].rect);

		// ROIの型をfindNearestの引数の型指定に合わせる
		Mat resizedRoi;
		resize(roi, resizedRoi, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

		Mat roiAsFloat;
		resizedRoi.convertTo(roiAsFloat, CV_32FC1);

		Mat roiAsFlattenedFloat = roiAsFloat.reshape(1, 1);

		Mat currentChar(0, 0, CV_32F);

		// 最近傍決定(input: roi, k=1, output: currentChar)
		kNearest->findNearest(roiAsFlattenedFloat, 1, currentChar);

		float currentCharAsFloat = (float)currentChar.at<float>(0, 0);

		// 認識結果文字を逐次後ろにくっつけてrecognizedStringを更新していく
		recognizedString += char(int(currentCharAsFloat));



	}

	// テスト画像(原画像)をimshowしてちゃんと正しいroiが囲めているかを確認
	imshow("texts to be tested", inputTestingImg);

	cout << "recognized texts: " << recognizedString << endl;

	waitKey(0);

	return 0;
}
