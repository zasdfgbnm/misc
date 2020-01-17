inline float operator[](float2 self, int i) {
    if (i == 0) {
        return self.x;
    } else {
        return self.y;
    }
}

int main() {
    float2 a;
    a[0];
    a[1];
}