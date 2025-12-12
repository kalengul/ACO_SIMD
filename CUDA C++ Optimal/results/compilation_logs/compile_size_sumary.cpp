#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <algorithm>
#include <dirent.h>
#include <regex>

// Структура для хранения информации о функции
struct FunctionInfo {
    std::string name;
    std::string demangled_name;
    int registers;
    int smem_bytes;
    int cmem_bank0;
    int cmem_bank2;
    int cmem_bank3;
    int stack_frame;
    int spill_stores;
    int spill_loads;
    std::string architecture;
};

// Структура для хранения параметров конфигурации из имени файла
struct ConfigParams {
    std::string filename;
    int graph_size;
    int register_count;
    std::string original_name;
};

// Функция для демангирования имен (упрощенная версия)
std::string demangle_name(const std::string& mangled) {
    // Простой демангинг для основных функций
    std::map<std::string, std::string> name_map = {
        {"_Z20parallelACOIteration", "parallelACOIteration"},
        {"_Z26updateAndComputePheromones", "updateAndComputePheromones"},
        {"_Z16updatePheromones", "updatePheromones"},
        {"_Z21antColonyOptimization", "antColonyOptimization"},
        {"_Z20computeProbabilities", "computeProbabilities"},
        {"_Z19initializeHashTable", "initializeHashTable"},
        {"__internal_trig_reduction_slowpathd", "internal_trig_reduction"}
    };
    
    for (const auto& pair : name_map) {
        if (mangled.find(pair.first) != std::string::npos) {
            return pair.second;
        }
    }
    return mangled;
}

// Функция для извлечения числа из строки
int extract_number(const std::string& str) {
    std::stringstream ss;
    for (char c : str) {
        if (isdigit(c)) ss << c;
    }
    int result;
    ss >> result;
    return result;
}

// Функция для извлечения параметров из имени файла
ConfigParams extract_params_from_filename(const std::string& filename) {
    ConfigParams params;
    params.filename = filename;
    params.original_name = filename.substr(0, filename.length() - 4); // убираем .txt
    
    // Используем регулярные выражения для извлечения параметров
    std::regex size_pattern("size_(\\d+)");
    std::regex reg_pattern("reg_(\\d+)");
    
    std::smatch match;
    
    // Извлекаем размер графа
    if (std::regex_search(filename, match, size_pattern) && match.size() > 1) {
        params.graph_size = std::stoi(match[1].str());
    } else {
        params.graph_size = -1; // значение по умолчанию если не найдено
    }
    
    // Извлекаем количество регистров
    if (std::regex_search(filename, match, reg_pattern) && match.size() > 1) {
        params.register_count = std::stoi(match[1].str());
    } else {
        params.register_count = -1; // значение по умолчанию если не найдено
    }
    
    return params;
}

// Функция для получения всех текстовых файлов в текущей директории
std::vector<std::string> get_text_files_in_directory() {
    std::vector<std::string> text_files;
    
    DIR *dir;
    struct dirent *ent;
    
    if ((dir = opendir(".")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            
            // Проверяем расширение файла
            if (filename.length() > 4 && 
                filename.substr(filename.length() - 4) == ".txt") {
                text_files.push_back(filename);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Could not open directory" << std::endl;
    }
    
    // Сортируем файлы по имени
    std::sort(text_files.begin(), text_files.end());
    
    return text_files;
}

// Функция для парсинга PTXAS вывода
std::vector<FunctionInfo> parse_ptxas_output(const std::string& filename) {
    std::vector<FunctionInfo> functions;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return functions;
    }
    
    FunctionInfo current_func;
    bool in_function = false;
    
    while (std::getline(file, line)) {
        // Убираем лишние пробелы
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) continue;
        
        // Парсим предупреждения
        if (line.find("ptxas warning") != std::string::npos) {
            std::cout << "WARNING: " << line << std::endl;
        }
        
        // Парсим информацию о глобальной памяти
        if (line.find("bytes gmem") != std::string::npos) {
            size_t pos = line.find("bytes gmem");
            if (pos != std::string::npos) {
                std::string gmem_str = line.substr(0, pos);
                int gmem = extract_number(gmem_str);
                std::cout << "Global memory: " << gmem << " bytes" << std::endl;
            }
        }
        
        // Начало информации о функции
        if (line.find("Compiling entry function") != std::string::npos) {
            if (in_function) {
                functions.push_back(current_func);
            }
            
            in_function = true;
            current_func = FunctionInfo();
            
            // Извлекаем имя функции и архитектуру
            size_t start = line.find("'_Z");
            size_t end = line.find("_'", start);
            if (start != std::string::npos && end != std::string::npos) {
                current_func.name = line.substr(start + 1, end - start - 1);
                current_func.demangled_name = demangle_name(current_func.name);
            }
            
            size_t arch_pos = line.find("'sm_");
            if (arch_pos != std::string::npos) {
                size_t arch_end = line.find("'", arch_pos + 1);
                current_func.architecture = line.substr(arch_pos + 1, arch_end - arch_pos - 1);
            }
        }
        
        // Парсим свойства функции
        else if (in_function && line.find("bytes stack frame") != std::string::npos) {
            current_func.stack_frame = extract_number(line);
            
            // Парсим spill memory
            if (line.find("spill stores") != std::string::npos) {
                size_t spill_pos = line.find("spill stores");
                if (spill_pos != std::string::npos) {
                    std::string spill_str = line.substr(0, spill_pos);
                    current_func.spill_stores = extract_number(spill_str);
                }
            }
            
            if (line.find("spill loads") != std::string::npos) {
                size_t spill_pos = line.find("spill loads");
                if (spill_pos != std::string::npos) {
                    std::string spill_str = line.substr(0, spill_pos);
                    current_func.spill_loads = extract_number(spill_str);
                }
            }
        }
        
        // Парсим использование ресурсов
        else if (in_function && line.find("Used") != std::string::npos) {
            // Регистры
            size_t reg_pos = line.find("registers");
            if (reg_pos != std::string::npos) {
                std::string reg_str = line.substr(line.find("Used") + 5, reg_pos - line.find("Used") - 5);
                current_func.registers = extract_number(reg_str);
            }
            
            // Shared memory
            size_t smem_pos = line.find("bytes smem");
            if (smem_pos != std::string::npos) {
                size_t smem_start = line.rfind(",", smem_pos);
                if (smem_start == std::string::npos) smem_start = line.find("Used") + 5;
                std::string smem_str = line.substr(smem_start + 1, smem_pos - smem_start - 1);
                current_func.smem_bytes = extract_number(smem_str);
            }
            
            // Constant memory bank 0
            size_t cmem0_pos = line.find("cmem[0]");
            if (cmem0_pos != std::string::npos) {
                size_t cmem0_start = line.rfind(",", cmem0_pos);
                if (cmem0_start == std::string::npos) cmem0_start = line.find("Used") + 5;
                std::string cmem0_str = line.substr(cmem0_start + 1, cmem0_pos - cmem0_start - 1);
                current_func.cmem_bank0 = extract_number(cmem0_str);
            }
            
            // Constant memory bank 2
            size_t cmem2_pos = line.find("cmem[2]");
            if (cmem2_pos != std::string::npos) {
                size_t cmem2_start = line.rfind(",", cmem2_pos);
                if (cmem2_start == std::string::npos) cmem2_start = line.find("Used") + 5;
                std::string cmem2_str = line.substr(cmem2_start + 1, cmem2_pos - cmem2_start - 1);
                current_func.cmem_bank2 = extract_number(cmem2_str);
            }
            
            // Constant memory bank 3
            size_t cmem3_pos = line.find("cmem[3]");
            if (cmem3_pos != std::string::npos) {
                size_t cmem3_start = line.rfind(",", cmem3_pos);
                if (cmem3_start == std::string::npos) cmem3_start = line.find("Used") + 5;
                std::string cmem3_str = line.substr(cmem3_start + 1, cmem3_pos - cmem3_start - 1);
                current_func.cmem_bank3 = extract_number(cmem3_str);
            }
            
            // Завершаем текущую функцию
            functions.push_back(current_func);
            in_function = false;
        }
    }
    
    // Добавляем последнюю функцию если нужно
    if (in_function) {
        functions.push_back(current_func);
    }
    
    file.close();
    return functions;
}

// Функция для сохранения в CSV файл
void save_to_csv(const std::vector<FunctionInfo>& functions, const std::string& output_filename, 
                 const ConfigParams& config_params) {
    std::ofstream outfile(output_filename, std::ios::app); // append mode
    
    // Записываем заголовок если файл пустой
    if (outfile.tellp() == 0) {
        outfile << "Config;GraphSize;RegisterCount;Function;DemangledName;Registers;SMem_Bytes;CMem_Bank0;CMem_Bank2;CMem_Bank3;"
                << "Stack_Frame;Spill_Stores;Spill_Loads;Architecture" << std::endl;
    }
    
    // Записываем данные для каждой функции
    for (const auto& func : functions) {
        outfile << config_params.original_name << ";"
                << config_params.graph_size << ";"
                << config_params.register_count << ";"
                << func.name << ";"
                << func.demangled_name << ";"
                << func.registers << ";"
                << func.smem_bytes << ";"
                << func.cmem_bank0 << ";"
                << func.cmem_bank2 << ";"
                << func.cmem_bank3 << ";"
                << func.stack_frame << ";"
                << func.spill_stores << ";"
                << func.spill_loads << ";"
                << func.architecture << std::endl;
    }
    
    outfile.close();
}

// Функция для создания сводного отчета
void create_summary_report(const std::vector<FunctionInfo>& functions, const std::string& summary_filename, 
                          const ConfigParams& config_params) {
    std::ofstream summary(summary_filename, std::ios::app);
    
    summary << "=== CONFIGURATION: " << config_params.original_name << " ===" << std::endl;
    summary << "Graph Size: " << (config_params.graph_size != -1 ? std::to_string(config_params.graph_size) : "N/A") << std::endl;
    summary << "Register Count: " << (config_params.register_count != -1 ? std::to_string(config_params.register_count) : "N/A") << std::endl;
    
    int total_registers = 0;
    int total_smem = 0;
    int total_cmem = 0;
    int max_registers = 0;
    std::string max_registers_func;
    
    for (const auto& func : functions) {
        total_registers += func.registers;
        total_smem += func.smem_bytes;
        total_cmem += func.cmem_bank0 + func.cmem_bank2 + func.cmem_bank3;
        
        if (func.registers > max_registers) {
            max_registers = func.registers;
            max_registers_func = func.demangled_name;
        }
    }
    
    summary << "Total functions: " << functions.size() << std::endl;
    summary << "Total registers used: " << total_registers << std::endl;
    summary << "Total shared memory: " << total_smem << " bytes" << std::endl;
    summary << "Total constant memory: " << total_cmem << " bytes" << std::endl;
    summary << "Function with most registers: " << max_registers_func << " (" << max_registers << " registers)" << std::endl;
    summary << "Functions with spill memory: ";
    
    bool has_spill = false;
    for (const auto& func : functions) {
        if (func.spill_stores > 0 || func.spill_loads > 0) {
            summary << func.demangled_name << " ";
            has_spill = true;
        }
    }
    if (!has_spill) summary << "None";
    summary << std::endl << std::endl;
    
    summary.close();
}

// Основная функция анализа
void analyze_ptxas_output(const std::string& input_filename, 
                         const std::string& output_csv, 
                         const std::string& summary_file,
                         const ConfigParams& config_params) {
    
    std::cout << "Analyzing PTXAS output from: " << input_filename << std::endl;
    std::cout << "Graph Size: " << config_params.graph_size << ", Register Count: " << config_params.register_count << std::endl;
    
    // Парсим вывод
    std::vector<FunctionInfo> functions = parse_ptxas_output(input_filename);
    
    if (functions.empty()) {
        std::cout << "Warning: No functions found in " << input_filename << std::endl;
        return;
    }
    
    std::cout << "Found " << functions.size() << " functions" << std::endl;
    
    // Сохраняем в CSV
    save_to_csv(functions, output_csv, config_params);
    
    // Создаем сводный отчет
    create_summary_report(functions, summary_file, config_params);
    
    std::cout << "Results saved to: " << output_csv << " and " << summary_file << std::endl;
    
    // Выводим краткую информацию в консоль
    std::cout << "\n=== QUICK OVERVIEW ===" << std::endl;
    for (const auto& func : functions) {
        std::cout << func.demangled_name << ": " << func.registers << " registers, " 
                  << func.smem_bytes << " bytes SMem" << std::endl;
    }
    std::cout << std::endl;
}
// Функция для создания сводной таблицы конфигураций
void create_configuration_summary(const std::vector<std::string>& text_files, const std::string& output_csv) {
    std::string config_summary_file = "configuration_summary.csv";
    std::ofstream summary(config_summary_file);
    
    summary << "Config;GraphSize;RegisterCount;FileName" << std::endl;
    
    for (const auto& file : text_files) {
        ConfigParams params = extract_params_from_filename(file);
        summary << params.original_name << ";"
                << params.graph_size << ";"
                << params.register_count << ";"
                << file << std::endl;
    }
    
    summary.close();
    std::cout << "Configuration summary saved to: " << config_summary_file << std::endl;
}
// Функция для автоматической обработки всех текстовых файлов в папке
void analyze_all_text_files_in_directory() {
    std::string output_csv = "gpu_resources_analysis.csv";
    std::string summary_file = "analysis_summary.txt";
    
    // Получаем все текстовые файлы в текущей директории
    std::vector<std::string> text_files = get_text_files_in_directory();
    
    if (text_files.empty()) {
        std::cout << "No .txt files found in current directory!" << std::endl;
        return;
    }
    
    std::cout << "Found " << text_files.size() << " text files:" << std::endl;
    for (const auto& file : text_files) {
        ConfigParams params = extract_params_from_filename(file);
        std::cout << "  - " << file << " (Size: " << params.graph_size << ", Reg: " << params.register_count << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Очищаем файлы при первом запуске
    std::ofstream clear_csv(output_csv);
    std::ofstream clear_summary(summary_file);
    clear_csv.close();
    clear_summary.close();
    
    // Обрабатываем каждый файл
    for (const auto& file : text_files) {
        ConfigParams config_params = extract_params_from_filename(file);
        
        std::cout << "=== Processing: " << file << " ===" << std::endl;
        analyze_ptxas_output(file, output_csv, summary_file, config_params);
    }
    
    std::cout << "\n=== BATCH ANALYSIS COMPLETE ===" << std::endl;
    std::cout << "Combined results saved to: " << output_csv << std::endl;
    std::cout << "Summary saved to: " << summary_file << std::endl;
    
    // Создаем дополнительный файл с общей статистикой по конфигурациям
    create_configuration_summary(text_files, output_csv);
}

// Пример использования
int main() {
    std::cout << "=== GPU Resources Analyzer ===" << std::endl;
    std::cout << "Automatically analyzing all .txt files in current directory..." << std::endl;
    std::cout << "Extracting graph size and register count from filenames..." << std::endl;
    
    // Автоматически анализируем все текстовые файлы в папке
    analyze_all_text_files_in_directory();

    return 0;
}