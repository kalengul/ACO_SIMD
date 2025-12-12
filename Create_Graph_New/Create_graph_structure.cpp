#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    // Шаблон повторяющейся структуры (21 строка)
    std::vector<std::string> templateStructure = {
        "-1.0 1.0 -100.0 -100.0 -100.0",
        "0.0 2.0 4.0 6.0 8.0",
        "0.0 1.0 -100.0 -100.0 -100.0",
        "0.0 0.2 0.4 0.6 0.8",
        "0.0 0.1 -100.0 -100.0 -100.0",
        "0.0 0.02 0.04 0.06 0.08",
        "0.0 0.01 -100.0 -100.0 -100.0",
        "0.0 0.002 0.004 0.006 0.008",
        "0.0 0.001 -100.0 -100.0 -100.0",
        "0.0 0.0002 0.0004 0.0006 0.0008",
        "0.0 0.0001 -100.0 -100.0 -100.0",
        "0.0 0.00002 0.00004 0.00006 0.00008",
        "0.0 0.00001 -100.0 -100.0 -100.0",
        "0.0 0.000002 0.000004 0.000006 0.000008",
        "0.0 0.000001 -100.0 -100.0 -100.0",
        "0.0 0.0000002 0.0000004 0.0000006 0.0000008",
        "0.0 0.0000001 -100.0 -100.0 -100.0",
        "0.0 0.00000002 0.00000004 0.00000006 0.00000008",
        "0.0 0.00000001 -100.0 -100.0 -100.0",
        "0.0 0.000000002 0.000000004 0.000000006 0.000000008",
        "0.0 0.000000001 -100.0 -100.0 -100.0"
    };

    // Запрос количества повторений у пользователя
    int numRepetitions;
    std::cout << "Kol Structure (21 layers): ";
    std::cin >> numRepetitions;

    // Запрос имени файла
    std::string filename;
    std::cout << "Output file: ";
    std::cin >> filename;

    // Создание и запись в файл
    std::ofstream outputFile(filename);
    
    if (!outputFile.is_open()) {
        std::cerr << "Error: " << filename << std::endl;
        return 1;
    }

    // Запись структур в файл
    for (int i = 0; i < numRepetitions; ++i) {
        for (const auto& line : templateStructure) {
            outputFile << line << std::endl;
        }
        
        // Добавляем пустую строку между структурами (опционально)
        if (i < numRepetitions - 1) {
            outputFile << std::endl;
        }
    }

    outputFile.close();
    
    std::cout << "Faile " << filename << " is created: " 
              << numRepetitions << " structure." << std::endl;
    std::cout << "Lines: " << numRepetitions * templateStructure.size() << std::endl;

    return 0;
}