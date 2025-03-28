#include "EventManager.h"
#include <iostream>

// ���������� ������ ���������� ������� � �������
void EventManager::addEvent(std::shared_ptr<Event> event) {
    eventQueue.push(event); // ��������� ������� � �������
}

// ���������� ������ ��������� ���������� �������
// EventManager::getInstance().processNextEvent(modelTime);
void EventManager::processNextEvent(double& modelTime) {
    // ���������, ���� �� ������� � �������
    if (eventQueue.empty()) {
        std::cout << "No events to process.\n"; // ���� ��� �������, ������� ���������
        return; // ������� �� ������
    }

    // �������� ��������� ������� �� ������� (� ��������� �����������)
    auto nextEvent = eventQueue.top();
    eventQueue.pop(); // ������� ��� �� �������

    // ������������ �������, ��������� ������� ����� ������
    nextEvent->processEvent(modelTime);
}

