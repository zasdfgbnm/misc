int main() {
    auto mylambda2 = [] __device__  (int param_a, int param_b) { return param_a + param_b; };
    mylambda2(1, 2, 3);
}