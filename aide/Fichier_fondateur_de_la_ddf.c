float* inverser(float * M, uint n) {
	float a[n*n]
	memcpy(a, M, sizeof(float)*n*n);
	float * _inverse = malloc(sizeof(float)*n*n);
	for (uint i=0; i < n; i++) for (uint j=0; j < n; j++) _inverse[i*n + j] = (i==j);
	float coef;
	for (uint L=0; L < n; L++) {
		for (uint y=0; y < n; y++) {
			if (y == L) continue;// #skip cette etape
			
			if (a[L*n + L] == 0) raise(SIGINT);

			coef = a[y*n + L]/a[L*n + L];
			
			for (uint k=0; k < n; k++) {
				a[y*n + k] -= a[L*n + k]*coef;
				_inverse[y*n + k] -= _inverse[L*n + k]*coef;
			}
		}
	}
	for (uint L=0; L < n; L++) {
		coef = a[L*n + L];
		if (coef == 0) raise(SIGINT);
			
		for (uint i=0; i < n; i++) {
			a[L*n + i] /= coef;
			_inverse[L*n + i] /= coef;
		}
	}
	return _inverse;
}
