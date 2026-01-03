#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <direct.h>

#define MAX_LINE_LENGTH 1024

// Функция для проверки, является ли строка числом
int is_number(const char* str) {
    if (str == NULL || *str == '\0') return 0;
    while (*str) {
        if (!isdigit((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

// Функция обновления файла
int update_file(const char* filename, const char* param_size, const char* graph_file, int is_main_file) {
    FILE* fp_in, * fp_out;
    char line[MAX_LINE_LENGTH];
    char temp_filename[MAX_LINE_LENGTH + 10];
    int updated_size = 0;
    int updated_graph = 0;
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
        fprintf(stderr, "Error: Cannot open temporary file %s for writing\n", temp_filename);
        fclose(fp_in);
        return 0;
    }

    // Читаем и обрабатываем каждую строку
    while (fgets(line, sizeof(line), fp_in) != NULL) {
        line_num++;

        // Проверяем строку на наличие PARAMETR_SIZE
        if (strstr(line, "#define PARAMETR_SIZE") != NULL) {
            fprintf(fp_out, "#define PARAMETR_SIZE %s\n", param_size);
            updated_size = 1;
            printf("  Updated PARAMETR_SIZE to %s in %s\n", param_size, filename);
        }
        // Проверяем строку на наличие NAME_FILE_GRAPH (только для main файла)
        else if (is_main_file && strstr(line, "#define NAME_FILE_GRAPH") != NULL) {
            fprintf(fp_out, "#define NAME_FILE_GRAPH \"%s\"\n", graph_file);
            updated_graph = 1;
            printf("  Updated NAME_FILE_GRAPH to \"%s\" in %s\n", graph_file, filename);
        }
        else {
            fputs(line, fp_out);
        }
    }

    fclose(fp_in);

    if (!updated_size) {
        // Если PARAMETR_SIZE не найден, добавляем в начало файла
        rewind(fp_out);
        FILE* fp_temp = fopen("temp_swap.tmp", "w");
        if (fp_temp) {
            fprintf(fp_temp, "#define PARAMETR_SIZE %s\n", param_size);
            if (is_main_file && !updated_graph) {
                fprintf(fp_temp, "#define NAME_FILE_GRAPH \"%s\"\n", graph_file);
            }

            fclose(fp_out);
            fp_out = fopen(temp_filename, "r");
            if (fp_out) {
                char buffer[256];
                while (fgets(buffer, sizeof(buffer), fp_out)) {
                    fputs(buffer, fp_temp);
                }
                fclose(fp_out);
            }
            fclose(fp_temp);
            remove(temp_filename);
            rename("temp_swap.tmp", temp_filename);
            fp_out = fopen(temp_filename, "a");
        }
    }

    if (fp_out) fclose(fp_out);

    // Заменяем оригинальный файл временным
    if (remove(filename) != 0) {
        fprintf(stderr, "Warning: Cannot remove original file %s, trying to copy...\n", filename);
    }

    if (rename(temp_filename, filename) != 0) {
        fprintf(stderr, "Error: Cannot rename temporary file to %s\n", filename);
        // Пытаемся скопировать содержимое
        fp_in = fopen(temp_filename, "r");
        fp_out = fopen(filename, "w");
        if (fp_in && fp_out) {
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), fp_in)) {
                fputs(buffer, fp_out);
            }
            fclose(fp_in);
            fclose(fp_out);
            remove(temp_filename);
        }
        else {
            if (fp_in) fclose(fp_in);
            if (fp_out) fclose(fp_out);
            return 0;
        }
    }

    return 1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: update_params.exe <PARAMETR_SIZE>\n");
        printf("Example: update_params.exe 336\n");
        printf("Current directory: %s\n", _getcwd(NULL, 0));
        return 1;
    }

    const char* param_size = argv[1];

    // Проверяем, что параметр - число
    if (!is_number(param_size)) {
        fprintf(stderr, "Error: PARAMETR_SIZE must be a positive integer\n");
        return 1;
    }

    printf("\n========================================\n");
    printf("Updating configuration files...\n");
    printf("PARAMETR_SIZE = %s\n", param_size);

    // Проверяем существование исходных файлов
    if (_access("main_omp.cpp", 0) != 0) {
        fprintf(stderr, "Error: main_omp.cpp not found!\n");
        return 1;
    }

    if (_access("cuda_module.cu", 0) != 0) {
        fprintf(stderr, "Error: cuda_module.cu not found!\n");
        return 1;
    }

    // Создаем имя файла графа
    char graph_filename[256];
    snprintf(graph_filename, sizeof(graph_filename),
        "Parametr_Graph/test%s_4.txt", param_size);

    printf("Graph file: %s\n", graph_filename);

    // Проверяем существует ли директория для графов
    if (_access("Parametr_Graph", 0) != 0) {
        printf("Warning: Parametr_Graph directory not found!\n");
        printf("Creating Parametr_Graph directory...\n");
        if (_mkdir("Parametr_Graph") != 0) {
            printf("Warning: Could not create Parametr_Graph directory\n");
        }
    }

    // Обновляем main_omp.cpp
    printf("\nUpdating main_omp.cpp...\n");
    if (!update_file("main_omp.cpp", param_size, graph_filename, 1)) {
        fprintf(stderr, "Error: Failed to update main_omp.cpp\n");
        return 1;
    }

    // Обновляем cuda_module.cu
    printf("\nUpdating cuda_module.cu...\n");
    if (!update_file("cuda_module.cu", param_size, "", 0)) {
        fprintf(stderr, "Error: Failed to update cuda_module.cu\n");
        return 1;
    }

    printf("\n========================================\n");
    printf("All files updated successfully!\n");
    printf("========================================\n\n");

    return 0;
}