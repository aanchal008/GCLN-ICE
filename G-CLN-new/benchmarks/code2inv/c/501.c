int main() {
  // variable declarations
  int x;
  int y;
// pre-conditions
  assume((y<=x));   
  (x = 0); // x = x + 1
  (y = 0); 
  
  while ((x < 200)) {
    {
        //x ,y
        if(((x%2)==0)){
          (y  = (y + 1));
        }
        // x1, y1
        (x = (x + 1));
        //x2, y2
    }
  }
  // post-condition
  assert( (y <= 2*x) );
}
// y<x ^ (x' = 0) ^ (y' = 2) => I(x',y')

// I(x,y) ^ (x < 1000) ^ (x%2 == 0) ^ (x1 = x + 1) ^ (y1 = y) ^ (x2 = x1) ^ (y2 = y1 + 1) => I(x2 , y2)
// I(x,y) ^ (x < 1000) ^ (x%2 != 0) ^ (x1 = x) ^ (y1 = y) ^ (y2 = y1 + 1) ^ (x2 = x1) => I(x2, y2)
//Post
// I(x,y) ^ (x >= 1000) => (y>x)

//for 501.b assert(8y >= x)
