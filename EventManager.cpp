#include "EventManager.h"
#include <iostream>

// Реализация метода добавления события в очередь
void EventManager::addEvent(std::shared_ptr<Event> event) {
    eventQueue.push(event); // Добавляем событие в очередь
}

// Реализация метода обработки следующего события
// EventManager::getInstance().processNextEvent(modelTime);
void EventManager::processNextEvent(double& modelTime) {
    // Проверяем, есть ли события в очереди
    if (eventQueue.empty()) {
        std::cout << "No events to process.\n"; // Если нет событий, выводим сообщение
        return; // Выходим из метода
    }

    // Получаем следующее событие из очереди (с наивысшим приоритетом)
    auto nextEvent = eventQueue.top();
    eventQueue.pop(); // Удаляем его из очереди

    // Обрабатываем событие, передавая текущее время модели
    nextEvent->processEvent(modelTime);
}

