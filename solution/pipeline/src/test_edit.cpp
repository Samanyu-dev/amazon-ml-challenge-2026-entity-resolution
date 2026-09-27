#define main engine_main
#include "retrieve.cpp"
#undef main
int main(){string a,b;while(getline(cin,a)&&getline(cin,b))cout<<edit(a,b)<<"\n";}
