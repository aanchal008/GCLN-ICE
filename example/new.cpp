#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    ofstream myFile;
    myFile.open("515.csv");
    int x = 0;
    int y = 0;

    while(x < 200)
    {
        y = y + 1;
        x = x + 3;
        myFile << 0 << "," << 0 << "," << 1 << "," << x << "," << y << "," << 1 << "\n";
    }
}