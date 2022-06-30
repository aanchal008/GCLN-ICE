int main() {
  // variable declarations    suppose the variable here are x_1 and y_1 which takes values 1 and 0 respectively
  int x;
  int y;
  // pre-conditions
  //x_1 = 1 and already given value that x = x_1
  //y_1 = 0 and already given value that y = y_1
  (x = 1); 
  (y = 0); 
  // loop body         
  //here the variables at the start of the loop are x_2 and y_2 and have values x_2 = x and y_2 = y 
  //always at the head of the loop the variable storing the updated values will be x_2 and y_2
  while ((y < 100)) {
    {
    (x  = (x + y));     // let update the value and store in x_3 i.e., x_3 = x_2 + y_2 
    (y  = (y + 1));     // y_3 = y_2 + 1
    }

  }
  // when the loop condition becomes false the values of x and y are stored in x_2 and y_2 respectively
  // post-condition
  assert( (y >= 100) );      //hence the condition is as follows (x_2 >= y_2)
}

/* 1) Pre(x,y)  => I(x', y')
   (x = 1) ^ (y = 0) ^ (x' = x) ^ (y' = y)  => (x' >= y')

2) I(x, y) ^ (y < 100000) ^ (x' = x + y) ^ (y' = y + 1)  => I(x', y')
   (x >= y) ^ (y < 100000) ^ (x' = x + y) ^ (y' = y + 1) => (x' >= y')

3) I(x,y) ^ (y >= 100000) => Post
   (x >= y) ^ (y >= 100000) => (x >= y)
*/
  