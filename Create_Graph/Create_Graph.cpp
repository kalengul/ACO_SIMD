#include <iostream>
#include <fstream>
#include <iomanip>

int main()
{
    // Открываем файл для записи
    std::ofstream outFile("test336_100.txt");
    if (!outFile)
    {
        std::cerr << "Error Open File!" << std::endl;
        return 1;
    }
    // Записываем значения 1, -1
    outFile << "-1.0 1.0 ";
    for (double i = 0.2; i <= 9.9; i += 0.1)
    {
        outFile << "-100.0 ";
    }
    outFile << std::endl;
    // Записываем значения от 0.0 до 9.9
    for (double i = 0.0; i <= 9.9; i += 0.1)
    {
        outFile << i << " ";
    }
    outFile <<  std::endl;
    // Записываем значения от 0.000 до 0.099
    for (double i = 0.0; i < 0.1; i += 0.001)
    {
        outFile << i << " ";
    }
    outFile << std::endl;
    // Устанавливаем фиксированное представление и точность
    outFile << std::fixed << std::setprecision(5);
    // Записываем значения от 0.00000 до 0.00099
    for (double i = 0.0; i < 0.001; i += 0.00001)
    {
        outFile << i << " ";
    }
    outFile << std::endl;

    // Устанавливаем фиксированное представление и точность
    outFile << std::fixed << std::setprecision(7);
    // Записываем значения от 0.000000 до 0.00000099
    for (double i = 0.0; i < 0.00001; i += 0.0000001)
    {
        outFile << i << " ";
    }
    outFile << std::endl;
    // Устанавливаем фиксированное представление и точность
    outFile << std::fixed << std::setprecision(9);
    // Записываем значения от 0.00000000 до 0.0000000099
    for (double i = 0.0; i < 0.0000001; i += 0.000000001)
    {
        outFile << i << " ";
    }
    outFile << std::endl;


    // Закрываем файл
    outFile.close();
    std::cout << "End!" << std::endl;

    return 0;
}
