#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

int main() {
    // Открываем файл для записи
    std::ofstream outFile("test42.txt");

    // Проверяем, был ли файл открыт успешно
    if (!outFile) {
        std::cerr << "Error Open File" << std::endl;
        return 1;
    }

    // Записываем первую строку
    outFile << std::fixed << std::setprecision(1);
    outFile << "-1.0 1.0 ";

    for (int i = 2; i < 100000; ++i) { //100 1000 10000 100000
        outFile << "-100.0 ";
    }
    outFile << std::endl;
    //10^-9
    
    // 10^-4
    outFile << std::fixed << std::setprecision(4);
    for (float i = 0.0; i < 10.0; i += 0.0001) {
        outFile << i << " ";
    }
    outFile << std::endl;

    // 10^-9
    outFile << std::fixed << std::setprecision(9);
    for (float i = 0.0; i < 0.0001; i += 0.000000001) {
        outFile << " " << i;
    }
    outFile << std::endl;

    /*
  // 10^-3
    outFile << std::fixed << std::setprecision(3);
    for (float i = 0.0; i < 10.0; i += 0.001) {
        outFile << i << " ";
    }
    outFile << std::endl;

    // 10^-7
    outFile << std::fixed << std::setprecision(7);
    for (float i = 0.0; i < 0.001; i += 0.0000001) {
        outFile << " " << i;
    }
    outFile << std::endl;

    // 10^-9
    outFile << std::fixed << std::setprecision(9);
    for (float i = 0.0; i < 0.0000001; i += 0.000000001) {
        outFile << " " << i;
    }
    for (int i = 100; i < 10000; ++i) {
        outFile << " -100.0";
    }
    outFile << std::endl;
    */
    /*
    // 10^-2
    outFile << std::fixed << std::setprecision(2);
    for (float i = 0.0; i < 10.0; i += 0.01) {
        outFile << i << " ";
    }
    outFile << std::endl;

    // 10^-5
    outFile << std::fixed << std::setprecision(5);
    for (float i = 0.0; i < 0.01; i += 0.00001) {
        outFile << " " << i;
    }
    outFile << std::endl;

    // 10^-8
    outFile << std::fixed << std::setprecision(8);
    for (float i = 0.0; i < 0.00001; i += 0.00000001) {
        outFile << " " << i;
    }
    outFile << std::endl;

    // 10^-9
    outFile << std::fixed << std::setprecision(9);
    for (float i = 0.0; i < 0.00000001; i += 0.000000001) {
        outFile << " " << i;
    }
    for (float i = 0.0; i < 0.000000001; i += 0.000000000001) {
        outFile << " -100.0";
    }
    outFile << std::endl;
    */
    /*
    // 10^-1
    outFile << std::fixed << std::setprecision(1);
    for (float i = 0.0; i < 10.0; i += 0.1) {
        outFile << i << " ";
    }
    outFile << std::endl;

    // 10^-3
    outFile << std::fixed << std::setprecision(3);
    for (float i = 0.0; i < 0.1; i += 0.001) {
        outFile << " " << i;
    }
    outFile << std::endl;

    // 10^-5
    outFile << std::fixed << std::setprecision(5);
    for (float i = 0.0; i < 0.001; i += 0.00001) {
        outFile << " " << i;
    }
    outFile << std::endl;

    // 10^-7
    outFile << std::fixed << std::setprecision(7);
    for (float i = 0.0; i < 0.000001; i += 0.0000001) {
        if (i == 0) outFile << "0.0000000";
        else outFile << " " << i;
    }
    // 10^-9
    outFile << std::fixed << std::setprecision(9);
    for (float i = 0.0; i < 0.0000001; i += 0.000000001) {
        outFile << " " << i;
    }
    */
    // Закрываем файл
    outFile.close();

    std::cout << "End" << std::endl;

    return 0;
}
