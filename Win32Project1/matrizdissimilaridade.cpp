#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

	//Configuracao, salva a configuracao do momento do jogo.
class configuracao
{
public:
	void setPos(int posi, char valu){
		pos[posi] = valu;
	}
	char getPos(int posi){
		return pos[posi];
	}
private:
	char pos[9];
};
	//Autoexplicativa
int getDissimilarity(configuracao c1,configuracao c2)
{
	int returnedvalue = 0;
	for (int a = 0; a < 8; a++){
		if (c1.getPos(a) != c2.getPos(a)){
			returnedvalue++;
		}
	}
	return returnedvalue;
}
	//Gera a matriz no arquivo tttdissimilarity.txt
bool gerarMatriz()
{
	vector<configuracao>	database;
	ifstream inputTTTDataset;
	ofstream outputTTTDissimilarity;
	string lineRead;
	configuracao newConfig;
	inputTTTDataset.open("tttdatabase.txt");


	if (inputTTTDataset.is_open()){
		while (!inputTTTDataset.eof())
		{
			getline(inputTTTDataset,lineRead);
			if (lineRead.size() != 0){
				for (int a = 0;
					a < 18;
					a += 2){
					newConfig.setPos(a / 2, lineRead.at(a));
				}
			}
			database.insert(database.end(), newConfig);
		}
	}
	else{
		return FALSE;
	}

	outputTTTDissimilarity.open("tttdissimilarity.txt");
	const int sizeOfMatrix = database.size();
	int ** dissimilarityMatrix = new int*[sizeOfMatrix];
	for (int i = 0; i < sizeOfMatrix; i++){
		dissimilarityMatrix[i] = new int[sizeOfMatrix];
	}

	outputTTTDissimilarity << "\	";
	for (int i = 0; i < sizeOfMatrix; i++){
		outputTTTDissimilarity << "X" << i << "	|";
	}
	outputTTTDissimilarity << "\n";
	for (int i = 0; i < sizeOfMatrix; i++){
		outputTTTDissimilarity << "X" << i << "	|";
		for (int j = 0; j < sizeOfMatrix; j++){
			dissimilarityMatrix[i][j] = getDissimilarity(database.at(i),database.at(j));
			outputTTTDissimilarity << dissimilarityMatrix[i][j] << "	|";
		}
		outputTTTDissimilarity << "\n";
	}


	return TRUE;
}