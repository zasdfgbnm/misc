struct A {
    int a;
    constexpr A(): a(0){};
};
__global__ void f(A *p) {
    __shared__ A s[5]; //error: initializer not allowed for __shared__ variable 
    s[0] = *p;
}
