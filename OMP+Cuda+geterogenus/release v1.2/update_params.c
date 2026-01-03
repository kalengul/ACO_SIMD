#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 1024
#define MAX_FILES 10

// Функция для проверки, является ли строка числом
int is_number(const char *str) {
    if (str == NULL || *str == '\0') return 0;
    while (*str) {
        if (!isdigit((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

// Функция обновления параметра в файле
int update_file(const char *filename, const char *param_name, const char *new_value, const char *graph_file) {
    FILE *fp_in, *fp_out;
    char line[MAX_LINE_LENGTH];
    char temp_filename[MAX_LINE_LENGTH + 10];
    int updated = 0;
    int line_num = 0;
    
    // Создаем имя временного файла
    snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", filename);
    
    // Открываем исходный файл
    fp_in = fopen(filename, "r");
    if (fp_in == NULL) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        return 0;
    }
    
    // Открываем временный файл для записи
    fp_out = fopen(temp_filename, "w");
    if (fp_out == NULL) {
        fprintf(stderr, "Error: Cannot open temporary file for writing\n");
        fclose(fp_in);
        return 0;
    }
    
    // Читаем и обрабатываем каждую строку
    while (fgets(line, sizeof(line), fp_in) != NULL) {
        line_num++;
        char *modified_line = line;
        
        // Ищем определение PARAMETR_SIZE в первых 30 строках
        if (line_num <= 30) {
            char *param_pos = strstr(line, "PARAMETR_SIZE");
            char *define_pos = strstr(line, "#define");
            
            if (param_pos != NULL && define_pos != NULL) {
                // Проверяем, что это действительно #define PARAMETR_SIZE
                char *after_define = define_pos + 7;
                while (*after_define && isspace((unsigned char)*after_define)) {
                    after_define++;
                }
                
                if (strstr(after_define, "PARAMETR_SIZE") == after_define) {
                    fprintf(fp_out, "#define PARAMETR_SIZE %s\n", new_value);
                    updated = 1;
                    modified_line = NULL; // Уже записали
                }
            }
            
            // Ищем определение NAME_FILE_GRAPH (только для main_omp.cpp)
            if (modified_line != NULL && strstr(line, "NAME_FILE_GRAPH") != NULL && 
                strstr(line, "#define") != NULL) {
                if (graph_file != NULL) {
                    fprintf(fp_out, "#define NAME_FILE_GRAPH \"%s\"\n", graph_file);
                    updated = 1;
                    modified_line = NULL; // Уже записали
                }
            }
        }
        
        // Записываем строку если не была изменена
        if (modified_line != NULL) {
            fputs(line, fp_out);
        }
    }
    
    fclose(fp_in);
    fclose(fp_out);
    
    // Заменяем оригинальный файл временным
    if (updated) {
        if (remove(filename) != 0) {
            fprintf(stderr, "Error: Cannot remove original file %s\n", filename);
            remove(temp_filename);
            return 0;
        }
        
        if (rename(temp_filename, filename) != 0) {
            fprintf(stderr, "Error: Cannot rename temporary file to %s\n", filename);
            return 0;
        }
        
        printf("Successfully updated %s with PARAMETR_SIZE=%s\n", filename, new_value);
        return 1;
    } else {
        // Если ничего не изменили, удаляем временный файл
        remove(temp_filename);
        fprintf(stderr, "Warning: PARAMETR_SIZE definition not found in first 30 lines of %s\n", filename);
        return 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: update_params.exe <PARAMETR_SIZE>\n");
        printf("Example: update_params.exe 336\n");
        return 1;
    }
    
    const char *param_size = argv[1];
    
    // Проверяем, что параметр - число
    if (!is_number(param_size)) {
        fprintf(stderr, "Error: PARAMETR_SIZE must be a positive integer\n");
        return 1;
    }
    
    // Создаем имя файла графа
    char graph_filename[256];
    snprintf(graph_filename, sizeof(graph_filename), 
             "Parametr_Graph/test%s_4.txt", param_size);
    
    printf("Updating files with PARAMETR_SIZE=%s\n", param_size);
    printf("Graph file: %s\n", graph_filename);
    
    // Обновляем main_omp.cpp
    if (!update_file("main_omp.cpp", "PARAMETR_SIZE", param_size, graph_filename)) {
        fprintf(stderr, "Error updating main_omp.cpp\n");
        return 1;
    }
    
    // Обновляем cuda_module.cu (без graph_filename)
    if (!update_file("cuda_module.cu", "PARAMETR_SIZE", param_size, NULL)) {
        fprintf(stderr, "Error updating cuda_module.cu\n");
        return 1;
    }
    
    printf("All files updated successfully!\n");
    return 0;
}