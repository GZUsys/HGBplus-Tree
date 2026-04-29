#include "HGB+-Tree.h"
#include "zipfian.h"
#include "zipfian_util.h"
#include <cassert>
#include <cmath>
// #include"pcm.h"
// #include "cpucounters.h" 
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <vector>
#include <string.h>
#include <stdint.h>
#include <string>
#include <thread>
#include <atomic>
#include <immintrin.h>
extern "C"
{
    #include <atomic_ops.h>
}  

typedef uint64_t            setkey_t;
typedef void*               setval_t;

#define MAX_KEY         ((uint64_t)(0x7fffffffffffffffULL))
#define DEFAULT_DURATION                5000
#define DEFAULT_INITIAL                 100
#define DEFAULT_NB_THREADS              1
#define DEFAULT_RANGE                   0x7FFFFFFF
#define DEFAULT_SEED                    0
#define DEFAULT_UPDATE                  100
#define DEFAULT_ALTERNATE               0
#define DEFAULT_EFFECTIV                0 
#define DEFAULT_UNBALANCED              0

#define THREADPOOL_NUM_CREATE           1
#define THREADPOOL_NUM_MAX              1

#define XSTR(s)                         STR(s)
#define STR(s)                          #s

#define VAL_MIN                         INT_MIN
#define VAL_MAX                         INT_MAX
#define DETECT_LATENCY
//#define UNIFORM
#define test_num   5000000
#define test_thread 10
int initial =     DEFAULT_INITIAL; 
unsigned int levelmax;
uint64_t records[50000000]={0};
uint64_t record[10][5000000]={0};
uint64_t latency, insert_nb[10] = {0},insert_nbs = 0;
__thread struct timespec T1[10], T2[10];
std::vector<uint64_t> buffer;
std::vector<uint64_t> sbuffer;
std::vector<long> slen;
std::vector<char> ops;
int worker_thread_num = 1; 
int mfencenum=0;
int clflushnum=0;
int splitnum=0;
int falsedelete = 0;
int falsefind = 0;
long bloom_might_insert = 0;

void read_data_from_file(char* file)
{
    long count = 0;

    FILE* fp = fopen(file, "r");
    if (fp == NULL) {
        exit(-1);
    }
    buffer.clear();
    printf("reading\n");
    while (1) {
        unsigned long long key;
        count = fscanf(fp, "%lld\n", &key);
        if (count != 1) {
            break;
        }
        buffer.push_back(key);
    }
    fclose(fp);
    printf("file closed\n");
}

void scan_data_from_file(char* file)
{
    long count = 0;

    FILE* fp = fopen(file, "r");
    if (fp == NULL) {
        exit(-1);
    }
    buffer.clear();
    ops.clear();
    printf("reading\n");
    while (1) {
        char str[100];
	    char * p;
        count = fscanf(fp, "%s\n",str);
        if (count != 1) {
            break;
        }
	    p=strtok(str,","); 
        buffer.push_back(atoll(p));
        // p=strtok(NULL,",");
        // ops.push_back(p[0]);
    }
    fclose(fp);
    printf("file closed\n");
}

void sd_data_from_file(char* file)
{
    long count = 0;

    FILE* fp = fopen(file, "r");
    if (fp == NULL) {
        exit(-1);
    }
    buffer.clear();
    slen.clear();
    printf("reading\n");
    while (1) {
        char str[100];
        char * p;
        count = fscanf(fp, "%s\n",str);
        if (count != 1) {
            break;
        }
        p=strtok(str,",");
        buffer.push_back(atoll(p));
        p=strtok(NULL,",");
        slen.push_back(atol(p));
    }
    fclose(fp);
    printf("file closed\n");
}

void clear_cache() {
	int size = 256 * 1024 * 1024;
	char *garbage = new char[size];
	for (int i = 0; i < size; ++i)
	    garbage[i] = i;
	for (int i = 100; i < size; ++i)
	    garbage[i] += garbage[i - 100];
	delete[] garbage;
}

void reset_all_logs(tree *t) {
    LOG *logs = (LOG *)t->log_area;
    for (int i = 0; i < worker_thread_num; i++) {
        logs[i].tid = 0;
        logs[i].head = 0;
    }
}

int main(int argc, char **argv)
{

	struct option long_options[] = {
        // These options don't set a flag
        {"help",                      no_argument,       NULL, 'h'},
        {"duration",                  required_argument, NULL, 'd'},
        {"initial-size",              required_argument, NULL, 'i'},
        {"thread-num",                required_argument, NULL, 't'},
        {"range",                     required_argument, NULL, 'r'},
        {"seed",                      required_argument, NULL, 'S'},
        {"update-rate",               required_argument, NULL, 'u'},
        {"unbalance",                 required_argument, NULL, 'U'},
        {"elasticity",                required_argument, NULL, 'x'},
        {NULL,                        0,                 NULL, 0  }
    };

    int i,c;
    long nb_threads = 1;
    while(1) {
        i = 0;
        c = getopt_long(argc, argv, "hAf:d:i:t:r:S:u:U:c:", long_options, &i);
        if(c == -1) break;
        if(c == 0 && long_options[i].flag == 0) c = long_options[i].val;
        switch(c) {
                case 0:
                    break;
                case 't':
                    nb_threads = atoi(optarg);
                    break;
                default:
                    exit(1);
        }
    }

    assert(nb_threads > 0);
    printf("Nb threads   : %d\n",  nb_threads);
	
    //false时为遍历，true时为二分
	bool methord = false;
	if(nb_threads >= 32){
		methord = true;
	}

    char loading_file[100];
    sprintf(loading_file, "%s", "/home/datafile/loadb.csv");
    read_data_from_file(loading_file);
    memset(record, 0, sizeof(record));
	worker_thread_num=nb_threads;//test_thread;
	int keys_per_thread=50000000/nb_threads;//test_num;

    tree* t = InitTree();
    initial=buffer.size();

    printf("buffer size: %d\n",initial);
    struct timeval start_time, end_time;
    uint64_t     time_interval;
  
  	std::thread threads[128];
//	 gettimeofday(&start_time, NULL);
	LOG *main_log = &(((LOG *)t->log_area)[0]);    // 单线程用第0块
	main_log->tid = pthread_self();                // 标记当前线程ID
	main_log->head = 0;
                for (int kt = 0; kt < 1; kt++) {
                    threads[kt] = std::thread([=]() {
                        int start = 0;
                        int end = 50000000;
                        for (int ii = start; ii < end; ii++) {
                            uint64_t kk = buffer[ii];
                            Insert_t(t,ii,kk,(char*)kk, main_log);
                        }
                    });
                }
                for (int t = 0; t < 1; t++) threads[t].join();

	/*
		gettimeofday(&end_time, NULL);
    time_interval = 1000000 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec;
    printf("delete time_interval = %lu ns\n", time_interval * 1000);
    printf("average delete op = %lu ns\n",    time_interval * 1000 / initial);
*/

printf("load time is clflushnum is %ld\n",clflushnum);
printf("load time is mfence is %ld\n",mfencenum);
printf("load time is splitnum is %ld\n",splitnum);
printf("tree height is %d\n",t->height);
clflushnum=0;mfencenum=0;splitnum=0;
printf("load is end\n");
			reset_all_logs(t);
			clear_cache();
            buffer.clear();
            sleep(5);
            sprintf(loading_file, "%s", "/home/datafile/insert50m.csv");
            scan_data_from_file(loading_file);
            // read_data_from_file(loading_file);
            // sd_data_from_file(loading_file);
            // pcm::PCM * m = pcm::PCM::getInstance();
    	    // auto status = m->program();
	    gettimeofday(&start_time, NULL);
        // printf("begin\n");

	    // pcm::SystemCounterState before_sstate = pcm::getSystemCounterState();
                for (int kt = 0; kt < worker_thread_num; kt++) {
                    threads[kt] = std::thread([=]() {   

						pthread_t tid = pthread_self();     
						LOG *mylog = occupyLog(t, tid);
						if (mylog == NULL) {
							printf("线程 %lu 无法获取日志空间，退出！\n", tid);
							return;
						}        
                        int start = keys_per_thread * kt;
                        int end = start+keys_per_thread;
                        for (int ii = start; ii < end; ii++) {
                            uint64_t kk = buffer[ii];
                            // char op = (char)ops[ii];
                            char op = 'i';
                            // unsigned long lend;
                            // lend = slen[ii]*100000000000;
                            
                            // char op;
                            // unsigned long lend = slen[ii];
                            // if(lend > 0){lend = lend*100000000000;op='s';}
                            // if(lend == 0){op='i';}


			switch (op)
                        {
                        case 'r':
                            // clock_gettime(CLOCK_MONOTONIC, &T1[kt]); 
                            if(methord){
                                Lookup_b(t, kk);
                            }else{
                                Lookup_t(t, kk);
                            }
                            // clock_gettime(CLOCK_MONOTONIC, &T2[kt]);
                            // latency = ((T2[kt].tv_sec - T1[kt].tv_sec) * 1000000000 + (T2[kt].tv_nsec - T1[kt].tv_nsec)) / 100;
                            // record[kt][latency] += 1;
                            // insert_nb[kt] += 1;
				            break;
                        case 'i':
				            // clock_gettime(CLOCK_MONOTONIC, &T1[kt]);
                            if(methord){
                                Insert_b(t, kk, (char*)kk, mylog);
                            }else{
                                Insert_t(t, ii, kk, (char*)kk, mylog);
                            }
                            // clock_gettime(CLOCK_MONOTONIC, &T2[kt]);
                            // latency = ((T2[kt].tv_sec - T1[kt].tv_sec) * 1000000000 + (T2[kt].tv_nsec - T1[kt].tv_nsec)) / 100;
                            // record[kt][latency] += 1;
                            // insert_nb[kt] += 1;                  
				break;
                        case 'd':
				            // clock_gettime(CLOCK_MONOTONIC, &T1[kt]);
                            Delete(t, kk);
                            // clock_gettime(CLOCK_MONOTONIC, &T2[kt]);
                            // latency = ((T2[kt].tv_sec - T1[kt].tv_sec) * 1000000000 + (T2[kt].tv_nsec - T1[kt].tv_nsec)) / 100;
                            // record[kt][latency] += 1;
                            // insert_nb[kt] += 1;
			    break;
                        case 's':
				            // clock_gettime(CLOCK_MONOTONIC, &T1[kt]);
                            // RangeLookUp(t, kk, kk + lend, methord);
                            // clock_gettime(CLOCK_MONOTONIC, &T2[kt]);
                            // latency = ((T2[kt].tv_sec - T1[kt].tv_sec) * 1000000000 + (T2[kt].tv_nsec - T1[kt].tv_nsec)) / 100;
                            // record[kt][latency] += 1;
                            // insert_nb[kt] += 1;
                break;
                        default :
                                printf("error\n");
                                break;
                        }

                        }
                        });
                }
                for (int t = 0; t < worker_thread_num; t++) threads[t].join();
    // pcm::SystemCounterState after_sstate = pcm::getSystemCounterState();
	// m->cleanup(); // 停止 PCM 统计
    gettimeofday(&end_time, NULL);
    // printf("tree height = %d\n",t->height);
	reset_all_logs(t);
    
    // printf("L3 misses: %lld\n",pcm::getL3CacheMisses(before_sstate, after_sstate));
	// printf("L3 Cache Hits: %lld\n",pcm::getL3CacheHits(before_sstate, after_sstate));
	// printf("L3 Cache Hit Ratio:%.2f %%\n",pcm::getL3CacheHitRatio(before_sstate, after_sstate) * 100);
	// printf("L3 MPKI: %.2f\n",pcm::getL3CacheMisses(before_sstate, after_sstate) / (getInstructionsRetired(before_sstate, after_sstate) / 1000.0));
	// printf("Instructions Retired: %lld\n", (long long)getInstructionsRetired(before_sstate, after_sstate));//总指令数
    // printf("DRAM Reads (bytes): %lld\n",pcm::getBytesReadFromMC(before_sstate, after_sstate));
    // printf("DRAM Writes (bytes): %lld\n",pcm::getBytesWrittenToMC(before_sstate, after_sstate));
    // printf("PM Reads (bytes): %lld\n",pcm::getBytesReadFromPMM(before_sstate, after_sstate));
    // printf("PM Writes (bytes): %lld\n",pcm::getBytesWrittenToPMM(before_sstate, after_sstate));
    
    time_interval = 1000000 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec;
    printf("time_interval = %lu ns\n", time_interval * 1000);
    printf("average op = %lu ns\n",    time_interval * 1000 / initial);

    double total_time_seconds = time_interval / 1000000.0;  // 将微秒转换为秒
    double throughput_ops = initial / total_time_seconds;
    double throughput_mops = throughput_ops / 1000000.0; // Mops/sec
    printf("Total operations: %lu\n", initial);
    printf("Total time: %lf seconds\n", total_time_seconds);
    printf("Throughput: %lf Mops/sec\n", throughput_mops);
  
    printf("clflush num  is   : %lld\n",             clflushnum);
    printf("mfence num is    : %lld\n",             mfencenum);
    printf("split num is    : %lld\n",             splitnum);

    //遍历所有线程的记录数据，将每个线程的记录累加到 records 数组中，并统计总的插入数量 insert_nbs
    
    //     for(int ik=0;ik<10;ik++)
    // {
    //     for(int jf=0;jf<5000000;jf++)
    //     {
    //        if(ik == 0 && record[ik][jf] != 0)
    //        {
    //          records[jf]=record[ik][jf];
    //        }
    //        if(ik > 0)
    //        {
    //           if(record[ik][jf] != 0 && records[jf]==0){records[jf]=record[ik][jf];}
    //           if(record[ik][jf] != 0 && records[jf]!=0){records[jf]=records[jf] + record[ik][jf];}
    //        }
    //     }
    //     insert_nbs=insert_nb[ik]+insert_nbs;
    // }


    //     uint64_t cnt = 0;
    //     uint64_t nb_min = insert_nbs * 0.1;
    //     uint64_t nb_50 = insert_nbs / 2;
    //     uint64_t nb_90 = insert_nbs * 0.9;
    //     uint64_t nb_99 = insert_nbs * 0.99;
    //     uint64_t nb_999 = insert_nbs * 0.999;
    //     uint64_t nb_9999 = insert_nbs * 0.9999;
    //     bool flag_50 = false, flag_90 = false, flag_99 = false,flag_min=false,flag_999 = false,flag_9999 = false;
    //     double latency_50, latency_90, latency_99, latency_min,latency_999,latency_9999;
    //     for (int i=0; i < 50000000 && !(flag_min && flag_50 && flag_90 && flag_99 &&flag_999 && flag_9999); i++){
    //         cnt += records[i];
    //         if (!flag_min && cnt >= nb_min){
    //             latency_min = (double)i / 10.0;
    //             flag_min = true;
    //         }
    //         if (!flag_50 && cnt >= nb_50){
    //             latency_50 = (double)i / 10.0;
    //             flag_50 = true;
    //         }
    //         if (!flag_90 && cnt >= nb_90){
    //             latency_90 = (double)i / 10.0;
    //             flag_90 = true;
    //         }
    //         if (!flag_99 && cnt >= nb_99){
    //             latency_99 = (double)i / 10.0;
    //             flag_99 = true;
    //         }
	//     if (!flag_999 && cnt >= nb_999){
    //             latency_999 = (double)i / 10.0;
    //             flag_999 = true;
    //         }
	//     if (!flag_9999 && cnt >= nb_9999){
    //             latency_9999 = (double)i / 10.0;
    //             flag_9999 = true;
    //         }
    //     }
    //     printf("min latency is %.1lfus\nmedium latency is %.1lfus\n90%% latency is %.1lfus\n99%% latency is %.1lfus\n,99%% latency is %.1lfus\n,99%% latency is %.1lfus\n", latency_min,latency_50, latency_90, latency_99,latency_999,latency_9999);
    
return 0;
    
}


