#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    ofstream myFile;
    myFile.open("501_new.csv");
    int x = 0;
    int y = 0;

    while(y < 100)
    {
      y = y + 1; 
      x = 2*y;

      myFile << 0 << "," << 0 <<"," << 1 << "," << x << "," << y << "," << '1' << "\n";   
        
    }
}