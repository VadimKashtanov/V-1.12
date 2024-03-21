#include "kernel/head/analyse/hessienne.cuh"

static __global__ void add_cudarray(float * arr, uint pos, float value) {
	arr[pos] += value;
};

static void changer_weight(Train_t * train, uint set, uint wpos, float value) {
	add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, set*train->mdl->weights + wpos, value);
	printf("");
};

void calculer_tableau_hessienne(Hessienne_t * hessienne) {
	Opti_t * opti = hessienne->opti;
	Train_t * train = opti->train;
	Mdl_t * mdl = train->mdl;

	uint wsize = mdl->weights;
	uint sets = train->sets;

	train_set_input(train);

	//float dfxy[sets], dfx[sets];
	float f[sets], fx[sets], fy[sets], fxy[sets];

	//tableau : x=w1 y=w0
	//	d (d/dw0) / dw1

	Progressbar0(20)

	uint y, x;

	for (uint w0=0; w0 < wsize; w0++) {
		//df(x)
		train_forward(train, 0);
		score_dloss(train);
		train_backward(train, 0);
		
		
		//df(x+)
	}

	/*for (uint w0=0; w0 < wsize; w0++) {
		for (uint w1=0; w1 < wsize; w1++) {
			y = w0;
			x = w1;
			//Hess = [d/dwxdwy]

			
			//( df/dx(y+1e-5) - df/dx(y) ) / 1e-5
			//( f(x+,y+) - f(y+) - f(x+) + f() ) / 1e-10
		
			//( forward/backward(y+1e-5) - forward/backward(y) ) / 1e-5

			//forward/backward(y+1e-5)
			//add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, set*wsize + w1, 1e-5);
			//CUDA_WAIT_KER()
			//train_forward(train, 0);
			//opti_dloss(hessienne->opti);
			//train_backward(train, 0);
			//SAFE_CUDA(cudaMemcpy(&dfxy, train->_meand + set*wsize + w0, sizeof(float)*1, cudaMemcpyDeviceToHost))

			//train_print_meands(train);

			//train_null_grad_meand(train);

			//forward/backward(y)
			//add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, set*wsize + w1, - 1e-5);
			//CUDA_WAIT_KER()
			//train_forward(train, 0);
			//opti_dloss(hessienne->opti);
			//train_backward(train, 0);
			//SAFE_CUDA(cudaMemcpy(&dfx, train->_meand + set*wsize + w0, sizeof(float)*1, cudaMemcpyDeviceToHost))

			//hessienne->tableau[set*wsize*wsize + w0*wsize + w1] = (dfxy - dfx)/1e-5;

			//train_null_grad_meand(train);

			//f()
			train_forward(train, 0);
			opti_loss(opti);
			for (uint i=0; i < sets; i++) f[i] = opti->set_score[i];

			//f(x+)
			for (uint i=0; i < sets; i++) {
				//add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, i*wsize + x, 1e-5);
				//CUDA_WAIT_KER()
				changer_weight(train, i, x, 1e-5);
			}
			train_forward(train, 0);
			opti_loss(opti);
			for (uint i=0; i < sets; i++) fx[i] = opti->set_score[i];

			//f(x+;y+)
			for (uint i=0; i < sets; i++) {
				//add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, i*wsize + y, 1e-5);
				//CUDA_WAIT_KER()
				changer_weight(train, i, y, 1e-5);
			}
			train_forward(train, 0);
			opti_loss(opti);
			for (uint i=0; i < sets; i++) fxy[i] = opti->set_score[i];

			//f(y+)
			for (uint i=0; i < sets; i++) {
				//add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, i*wsize + x, - 1e-5);
				//CUDA_WAIT_KER()
				changer_weight(train, i, x, - 1e-5);
			}
			train_forward(train, 0);
			opti_loss(opti);
			for (uint i=0; i < sets; i++) fy[i] = opti->set_score[i];

			//
			for (uint i=0; i < sets; i++) {
				//add_cudarray<<<dim3(1),dim3(1)>>>(train->_weight, i*wsize + y, - 1e-5);
				//CUDA_WAIT_KER()
				changer_weight(train, i, y, -1e-5);
			}

			//
			for (uint i=0; i < sets; i++) {
				//printf("%f %f %f %f\n", fxy[i], fy[i], fx[i], f[i]);
				hessienne->tableau[i*wsize*wsize + w0*wsize + w1] = (fxy[i] - fy[i] - fx[i] + f[i])/1e-10;
			}

			Progressbar((w0*wsize + w1)/(wsize*wsize), 20)
		}
	}*/

	Progressbar1(20)
}