#include "backsubstitute.h"

void * update_state_using_previous_layers(void *args){
	nn_thread_t * data = (nn_thread_t *)args;
	elina_manager_t * man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t * pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	int k;
	
	neuron_t ** out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	
	for(i=idx_start; i < idx_end; i++){
		bool already_computed= false;
		expr_t *lexpr = copy_expr(out_neurons[i]->lexpr);
		expr_t *uexpr = copy_expr(out_neurons[i]->uexpr);
		out_neurons[i]->lb = get_lb_using_previous_layers(man, fp, &lexpr, layerno);
		out_neurons[i]->ub = get_ub_using_previous_layers(man, fp, &uexpr, layerno);
		out_neurons[i]->backsubstituted_lexpr = lexpr;
		out_neurons[i]->backsubstituted_uexpr = uexpr;
		//free_expr(lexpr);
		//free_expr(uexpr);
		
	}
	return NULL;
}


void update_state_using_previous_layers_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno){
	//size_t NUM_THREADS = get_nprocs();
  	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t num_out_neurons = fp->layers[layerno]->dims;
	size_t i;
	if(num_out_neurons < NUM_THREADS){
		for (i = 0; i < num_out_neurons; i++){
	    		args[i].start = i; 
	    		args[i].end = i+1;   
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].linexpr0 = NULL;
			args[i].res = NULL;
	    		pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
			
	  	}
		for (i = 0; i < num_out_neurons; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	else{
		size_t idx_start = 0;
		size_t idx_n = num_out_neurons / NUM_THREADS;
		size_t idx_end = idx_start + idx_n;
		
		
	  	for (i = 0; i < NUM_THREADS; i++){
	    		args[i].start = idx_start; 
	    		args[i].end = idx_end;   
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].linexpr0 = NULL;
			args[i].res = NULL;
	    		pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
			idx_start = idx_end;
			idx_end = idx_start + idx_n;
	    		if(idx_end>num_out_neurons){
				idx_end = num_out_neurons;
			}
			if((i==NUM_THREADS-2)){
				idx_end = num_out_neurons;
				
			}
	  	}
		for (i = 0; i < NUM_THREADS; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
}


// additional function for run_deeppoly() for GPUARENA

void * update_state_layer_by_layer_lb(void *args)
{
	nn_thread_t *data = (nn_thread_t *)args;
	elina_manager_t *man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	int k = data->k;
	neuron_t **out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	for (i = idx_start; i < idx_end; i++)
	{
		bool already_computed = false;
		out_neurons[i]->lb = fmin(out_neurons[i]->lb, get_lb_using_prev_layer(man, fp, &out_neurons[i]->backsubstituted_lexpr, k));
		// printf("lower bound is %.6f\n", out_neurons[i]->lb);
	}
	return NULL;
}

void *update_state_layer_by_layer_ub(void *args)
{
	nn_thread_t *data = (nn_thread_t *)args;
	elina_manager_t *man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	int k = data->k;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	neuron_t **out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	for (i = idx_start; i < idx_end; i++)
	{
		bool already_computed = false;
		out_neurons[i]->ub = fmin(out_neurons[i]->ub, get_ub_using_prev_layer(man, fp, &out_neurons[i]->backsubstituted_uexpr, k));
		// printf("upper bound is %.6f\n", out_neurons[i]->ub);
	}
	return NULL;
}

void update_state_layer_by_layer_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno)
{
	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t num_out_neurons = fp->layers[layerno]->dims;
	size_t i;
	int k;
	if (fp->numlayers == layerno)
	{
		k = layerno - 1;
	}
	else if ((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors == 2))
	{
		k = layerno;
	}
	else
	{
		k = fp->layers[layerno]->predecessors[0] - 1;
	}
	while (k >= -1)
	{
		if (num_out_neurons < NUM_THREADS)
		{
			for (i = 0; i < num_out_neurons; i++)
			{
				args[i].start = i;
				args[i].end = i + 1;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_lb, (void *)&args[i]);
			}
			for (i = 0; i < num_out_neurons; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		else
		{
			size_t idx_start = 0;
			size_t idx_n = num_out_neurons / NUM_THREADS;
			size_t idx_end = idx_start + idx_n;
			for (i = 0; i < NUM_THREADS; i++)
			{
				args[i].start = idx_start;
				args[i].end = idx_end;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_lb, (void *)&args[i]);
				idx_start = idx_end;
				idx_end = idx_start + idx_n;
				if (idx_end > num_out_neurons)
				{
					idx_end = num_out_neurons;
				}
				if ((i == NUM_THREADS - 2))
				{
					idx_end = num_out_neurons;
				}
			}
			for (i = 0; i < NUM_THREADS; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		if (num_out_neurons < NUM_THREADS)
		{
			for (i = 0; i < num_out_neurons; i++)
			{
				args[i].start = i;
				args[i].end = i + 1;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_ub, (void *)&args[i]);
			}
			for (i = 0; i < num_out_neurons; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		else
		{
			size_t idx_start = 0;
			size_t idx_n = num_out_neurons / NUM_THREADS;
			size_t idx_end = idx_start + idx_n;
			for (i = 0; i < NUM_THREADS; i++)
			{
				args[i].start = idx_start;
				args[i].end = idx_end;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_ub, (void *)&args[i]);
				idx_start = idx_end;
				idx_end = idx_start + idx_n;
				if (idx_end > num_out_neurons)
				{
					idx_end = num_out_neurons;
				}
				if ((i == NUM_THREADS - 2))
				{
					idx_end = num_out_neurons;
				}
			}
			for (i = 0; i < NUM_THREADS; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		if (k < 0)
			break;
		else{
			k = fp->layers[k]->predecessors[0] - 1;
		}
	}
}

//end of my new functions