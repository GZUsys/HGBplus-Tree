#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
// #include <x86intrin.h>
// #include <malloc.h>
#include <stdint.h>
#include <time.h>
// #include <stdbool.h>
// #include <emmintrin.h>
#include <libpmemobj.h>
#include <sys/stat.h>
#include <atomic>
#include <vector>
#include <algorithm> // for std::fill
#include <array>
#include <pthread.h>

#define mfence() asm volatile("mfence":::"memory")
#define bitScan(x)  __builtin_ffs(x)
#define countBit(x) __builtin_popcount(x)

#define NODESIZE 512
#define CACHE_LINE_SIZE 64 
#define HALF_SIZE 15
#define NODE_SIZE 30
#define WRITER_BUFFER_INDEX 27
#define MID_NODE_SIZE 15
#define BITMAP_FULL_MASK  0x3fffffff
#define LOG_AREA_SIZE 20

#define entry_key_t int64_t
int split_num = 0;

extern char *start_addr;
extern char *curr_addr;

extern int worker_thread_num;
extern int mfencenum;
extern int clflushnum;
extern int splitnum;
extern int falsedelete;
extern int falsefind;

typedef struct tree tree;
typedef struct entry entry;
typedef struct innerNode innerNode;
typedef struct header header;
typedef struct leafNode leafNode;

struct entry{
	unsigned long key;
	void *ptr;
};

typedef struct{
  pthread_t tid;
  int head;
  int count;
  struct entry entries[LOG_AREA_SIZE];
} LOG;

struct tree{
	int height;
	void *root;
	LOG *log_area; 
};

struct innerNode
{
	uint32_t num;                      
    std::atomic<uint32_t> version;  
    struct innerNode *nextPtr;               
    void *leftmostPtr;           
	char unused[8];          
	struct entry entries[NODE_SIZE];
};

typedef struct {
    uint64_t bits;
} BloomFilter;

struct header{
    uint8_t  sort : 3;
    uint8_t  bitmap : 3;
    uint8_t  deleted : 1;
	uint8_t  min : 1;
    uint8_t  left_index;
    uint8_t  right_index;
    uint8_t  sort_index;
    std::atomic<uint32_t> version;
    BloomFilter filter;
};

struct leafNode{
    struct entry entries[NODE_SIZE];
    struct header hdr;
    leafNode *nextPtr;
    uint32_t left_lock;
    uint32_t right_lock;
};

POBJ_LAYOUT_BEGIN(Pbtree);
POBJ_LAYOUT_TOID(Pbtree, leafNode);
POBJ_LAYOUT_TOID(Pbtree, innerNode);
POBJ_LAYOUT_END(Pbtree);
PMEMobjpool *pop;

pthread_mutex_t LOCK_LOG;

void add_entry_to_log(LOG *mylog, entry *entries, int index, int count);
innerNode *allocInnerNode();
leafNode *allocLeafNode();
int file_exists(const char *filename);
void openPmemobjPool();
tree *InitTree(int threadNum);
leafNode *FindLeafNode_b(tree *t, unsigned long key);
leafNode *FindLeafNode_t(tree *t, unsigned long key);
int SearchInner(innerNode *node, unsigned long key);
int SearchSort(entry entries[], unsigned long key, int left, int right);
void SortLnode(entry entry[], int start, int end, int pos[]);
void QuickSort(entry *a, int low, int high);
void SelectSort(entry entries[] , int n);
void *Lookup_b(tree *t, unsigned long key);
void *Lookup_t(tree *t, unsigned long key);
void RangeLookUp(tree *t, unsigned long startKey, unsigned long endKey, bool methord);
void Insert_b(tree *t, unsigned long key, void *value, LOG *mylog);
void Insert_t(tree *t, uint64_t ii, unsigned long key, void *value, LOG *mylog);
void DirectInsertLeaf(tree *t, leafNode *node, unsigned long key, void *value, LOG *mylog);
void SortLeft(tree *t, leafNode *node, LOG *mylog);
void SortRight(tree *t, leafNode *node, LOG *mylog);
leafNode *SplitLeaf(leafNode *node);
int GetMidLocation(entry entries[], unsigned long key, int left, int right);
void AddInode(tree *t, void *currentNode, unsigned long key, void *splitNode, unsigned int height, bool methord);
void InodeSplit(tree *t, innerNode *iNode, unsigned long key, void *node, unsigned int height, bool methord);
void Delete(tree *t, unsigned long key);
void DelINode(tree *t, unsigned long key, int height);
void printLeaf(leafNode *node);
void printInnerNode(innerNode *iNode);

static inline unsigned long ffz(unsigned long word)
{
	asm("rep; bsf %1,%0"
		: "=r" (word)
		: "r" (~word));
	return word;
}

static inline void asm_clflush(volatile uint64_t *addr)
{
    asm volatile(".byte 0x66; clflush %0" : "+m" (*(volatile char *)(addr)));
}

/****************************************************************************************************************************/
uint8_t hash1(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key & 63;  
}

uint8_t hash2(uint64_t key) {
    key ^= key >> 31;
    key *= 0x9e3779b97f4a7c15ULL;  
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 30;
    return key & 63;
}

static inline int hash_tid(pthread_t tid, int mod) {
    uint64_t x = (uint64_t)tid;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;

    return x % mod;
}

void bloom_insert(BloomFilter  *filter, uint64_t key) {
    uint8_t h1 = hash1(key);
    uint8_t h2 = hash2(key);
    filter->bits |= ((uint64_t)1 << h1);
    filter->bits |= ((uint64_t)1 << h2);
}

int bloom_maybe_contains(BloomFilter  *filter, uint64_t key) {
    uint8_t h1 = hash1(key);
    uint8_t h2 = hash2(key);
    return ((filter->bits & ((uint64_t)1 << h1)) &&
            (filter->bits & ((uint64_t)1 << h2)));
}

int lazyfield_count(uint8_t bitmap) 
{
	return countBit(bitmap);
}

int comparebitmap(uint8_t bitmap, int pos)
{
    if (bitmap & (1UL << pos))
    {
        return 1;
    }
    return 0;
}

void clflush(void *data, int len)
{
	volatile char *ptr = (char *)((unsigned long)data &~(CACHE_LINE_SIZE-1));
	for(; ptr<data+len; ptr+=CACHE_LINE_SIZE){
		asm volatile(".byte 0x66; clflush %0" : "+m" (*((volatile char *)ptr)));
		// clflushnum++;
	}
	mfence();
	// mfencenum++;
}

//flush key from end to begin
void insclflushmore(void* end, void* begin)
{
    volatile char* bptr = (char*)((unsigned long)begin & ~(CACHE_LINE_SIZE - 1));
    volatile char* eptr = (char*)((unsigned long)end & ~(CACHE_LINE_SIZE - 1));
    for (; eptr >= bptr; eptr = eptr - CACHE_LINE_SIZE) {
        asm volatile(".byte 0x66; clflush %0" : "+m" (*((volatile char*)eptr)));
		// clflushnum++;
    }
    mfence();
	// mfencenum++;
}

//flush key from begin to end
void delclflushmore(void* begin, void* end)
{
    volatile char* bptr = (char*)((unsigned long)begin & ~(CACHE_LINE_SIZE - 1));
    volatile char* eptr = (char*)((unsigned long)end & ~(CACHE_LINE_SIZE - 1));
    for (; bptr <= eptr; bptr += CACHE_LINE_SIZE) {
        asm volatile(".byte 0x66; clflush %0" : "+m" (*((volatile char*)bptr)));
		// clflushnum++;
    }
    mfence();
	// mfencenum++;
}

void add_entry_to_log(LOG *mylog, entry *entries, int index, int count) {
    int start_pos = mylog->head;

    for (int i = index; i < count; i++) {
        mylog->entries[mylog->head] = entries[i];
        mylog->head = (mylog->head + 1) % LOG_AREA_SIZE;
    }

    if (start_pos + count <= LOG_AREA_SIZE) {
		clflush(&(mylog->entries[start_pos]), sizeof(entry) * count);
    } else {
        int first_part = LOG_AREA_SIZE - start_pos;
        int second_part = count - first_part;
		clflush(&(mylog->entries[start_pos]), sizeof(entry) * first_part);
		clflush(&(mylog->entries[0]), sizeof(entry) * second_part);
    }
}

void setRoot(tree *t, innerNode *newRoot){
    t->root = newRoot;
    t->height++;
	clflush(t, CACHE_LINE_SIZE);
}

void setNewRoot(innerNode *newRoot, void *currentNode, unsigned long key, void *splitNode){
    newRoot->leftmostPtr = currentNode; 
    newRoot->entries[0].key = key;
    newRoot->entries[0].ptr = splitNode;
    newRoot->num = 1;
	clflush(newRoot, CACHE_LINE_SIZE);
}

unsigned int getlock(std::atomic<uint32_t>* version) {
    std::atomic_thread_fence(std::memory_order_acquire);
    uint32_t ver = version->load(std::memory_order_acquire);  
    return ver;
}

uint32_t vlock(std::atomic<uint32_t>* version) {
    std::atomic_thread_fence(std::memory_order_acquire);
    uint32_t ver = version->load(std::memory_order_acquire);  
    if (ver < 2) {
        uint32_t new_ver = 3;
        if (version->compare_exchange_weak(ver, new_ver)) {
            return new_ver;
        }
        return 0;
    }
    else {
        if ((ver & 1) == 0 && version->compare_exchange_weak(ver, ver + 1)) {
            return ver;
        }
    }
    return 0;
}

void vunlock(std::atomic<uint32_t>* version) {
    version->fetch_add(1, std::memory_order_release);
}

uint32_t lockstatus(std::atomic<uint32_t>* version) {
  return version->load(std::memory_order_acquire) & 1;
}

LOG *occupyLog(tree *t, pthread_t TID) {
    pthread_mutex_lock(&LOCK_LOG);
    LOG *logs = t->log_area;
    int hash = hash_tid(TID, worker_thread_num);

    if (logs[hash].tid == 0) {
        logs[hash].tid = TID;
        logs[hash].head = 0;
		logs[hash].count = 0;
        pthread_mutex_unlock(&LOCK_LOG);
        return &logs[hash];
    }

    for (int i = (hash + 1) % worker_thread_num; i != hash; i = (i + 1) % worker_thread_num) {
        if (logs[i].tid == 0) {
            logs[i].tid = TID;
            logs[i].count = 0;
            pthread_mutex_unlock(&LOCK_LOG);
            return &logs[i];
        }
    }

    pthread_mutex_unlock(&LOCK_LOG);
    return NULL;
}

innerNode *allocInnerNode(){
	TOID(innerNode) p;
    POBJ_ZNEW(pop, &p, innerNode);
    innerNode *node = (innerNode *)pmemobj_direct(p.oid);
	node->version.store(2, std::memory_order_release);
	node->leftmostPtr = NULL;
	node->nextPtr = NULL;
	node->num = 0;
	return node;
}

leafNode *allocLeafNode(){
    TOID(leafNode) p;
    POBJ_ZNEW(pop, &p, leafNode);
    leafNode *n = (leafNode *)pmemobj_direct(p.oid); 
	n->hdr.version.store(2, std::memory_order_release);
    n->hdr.bitmap = 0;
	n->hdr.deleted = 0;
	n->hdr.min = 0;
    n->hdr.left_index = 0;
    n->hdr.right_index = HALF_SIZE;
    n->hdr.sort_index = HALF_SIZE;
	n->hdr.sort = 0;
    n->hdr.filter.bits = 0;
	n->nextPtr = NULL;
	n->left_lock = 0;
	n->right_lock = 0;
    return n;
}

int file_exists(const char *filename) {
	struct stat buffer;
	return stat(filename, &buffer);
}

void openPmemobjPool() {
	printf("use pmdk!\n");
	char pathname[100] = "/pmem0/HGBTree/pool";
	int sds_write_value = 0;
	pmemobj_ctl_set(NULL, "sds.at_create", &sds_write_value);
	if (file_exists(pathname) != 0) {
		printf("create new one.\n");
		if ((pop = pmemobj_create(pathname, POBJ_LAYOUT_NAME(Pbtree),
								(uint64_t)100ULL * 1024ULL * 1024ULL * 1024ULL, 0666)) == NULL) {
		perror("failed to create pool.\n");
		return;
		}
	} else {
		printf("open existing one.\n");
		if ((pop = pmemobj_open(pathname, POBJ_LAYOUT_NAME(Pbtree))) == NULL) {
		perror("failed to open pool.\n");
		return;
		}
	}
}

tree *InitTree(){
	openPmemobjPool();
	tree *t =(tree *)malloc(sizeof(tree)); 
	t->root = allocLeafNode(); 
	t->height = 1;
	printf("worker_thread_num = %d\n",worker_thread_num);
	t->log_area = (LOG *)malloc(sizeof(LOG) * worker_thread_num);
	memset(t->log_area, 0, sizeof(LOG) * worker_thread_num);
	pthread_mutex_init(&LOCK_LOG, NULL);
	return t;
}


leafNode *FindLeafNode_b(tree *t, unsigned long key){
	innerNode *iNode;
    int i, m, b, level, version;
	short num;
    long r;
restart:
    iNode = (innerNode *)t->root;
	level = t->height;
    for (i = level; i > 1; i--)
    {
		version = getlock(&iNode->version);
        if (lockstatus(&iNode->version))
        {
            goto restart;
        }
        num = iNode->num - 1;
        b = 0;
        while (b + 2 <= num) {
            m = (b + num) >> 1;
            r = key - iNode->entries[m].key;
            if (r > 0){
				b = m + 1;
			}
            else if (r < 0){
				num = m - 1;
			}
            else {
				if (version != getlock(&iNode->version)){
                    goto restart;
                }
				iNode = (innerNode *)iNode->entries[m].ptr;
                goto inner_done;
            }
        }
        for (; b <= num; b++)
        {
            if (key < iNode->entries[b].key) break;
        }
        if (b == 0)
        {
			if (version != getlock(&iNode->version))
            {
                goto restart;
            }
			iNode = (innerNode *)iNode->leftmostPtr;
        }
        else
        {
			if (version != getlock(&iNode->version))
            {
                goto restart;
            }
			iNode = (innerNode *)iNode->entries[b-1].ptr;
        }
    inner_done:;
    }
	return (leafNode *)iNode;
}

leafNode *FindLeafNode_t(tree *t, unsigned long key){
	innerNode *iNode;
    int i, j, level, version;
	short num;
    void *ret;
restart:
    iNode = (innerNode *)t->root;
	level = t->height;
    for (i = level; i > 1; i--)
    {
		version = getlock(&iNode->version);
        if (lockstatus(&iNode->version))
        {
            goto restart;
        }
        num = iNode->num;
		if(key < iNode->entries[0].key){
			if (version != getlock(&iNode->version))
            {
                goto restart;
            }
			iNode = (innerNode *)iNode->leftmostPtr;
			continue;
		}
        for(j = 1; j < num; j++){
			if(key < iNode->entries[j].key){
				if (version != getlock(&iNode->version))
				{
					goto restart;
				}
				iNode = (innerNode *)iNode->entries[j - 1].ptr;
				break;
			}
		}
		if(j == num){
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->entries[j - 1].ptr;
		}
		
    }
	return (leafNode *)iNode;
}

innerNode *FindInnerNode_b(tree *t, unsigned long key, unsigned int height){
	innerNode *iNode;
	int i, b, m, num, level;
	long r;
restart:
	iNode = (innerNode *)t->root;
    level = height;
    for (; level > 2; level--)
    {
		int version = getlock(&iNode->version);
		if (lockstatus(&iNode->version))
        {
            goto restart;
        } 
        num = iNode->num - 1;     
        b = 0;
        while (b + 2 <= num) {
            m = (b + num) >> 1;
            r = key - iNode->entries[m].key;
            if (r > 0) b = m + 1;
            else if (r < 0) num = m - 1;
            else {  
				if (version != getlock(&iNode->version))
                {
                    goto restart;
                }
				iNode = (innerNode *)iNode->entries[m].ptr;     
                goto inner_done;
            }
        }
        for (; b <= num; b++)
        {
			if (key < iNode->entries[b].key) break;
        }
        if (b == 0)
        {
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->leftmostPtr;
        }
        else
        {
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->entries[b-1].ptr;
        }                    
    inner_done:;
    }
	return iNode;
}

innerNode *FindInnerNode_t(tree *t, unsigned long key, unsigned int height){
	innerNode *iNode;
	int i, b, m, num, level;
	long r;
restart:
	iNode = (innerNode *)t->root;
    level = height;
    for (; level > 2; level--)
    {
		int version = getlock(&iNode->version);
		if (lockstatus(&iNode->version))
        {
            goto restart;
        } 
		num = iNode->num;
		if(key < iNode->entries[0].key){
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->leftmostPtr;
			continue;
		}
        for(i = 1; i < num; i++){
			if(key < iNode->entries[i].key){
				if (version != getlock(&iNode->version))
                {
                    goto restart;
                }
				iNode = (innerNode *)iNode->entries[i - 1].ptr;
				break;
			}
		}
		if(i == num){
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->entries[i - 1].ptr;
		}                 
    inner_done:;
    }
	return iNode;
}

int SearchInner(innerNode *node, unsigned long key){
	int low = 0, mid = 0, n = node->num, high = node->num - 1;

	while (low <= high) {
		mid = low + (high - low) / 2;
		if (node->entries[mid].key > key){
			high = mid - 1;
		}else if (node->entries[mid].key < key){
			low = mid + 1;
		}else
			return mid; 
    }
    if (low < n) {
        return low;
    }

    return -1;
}

int SearchSort(entry entries[], unsigned long key, int left, int right){
    int mid = 0, n = right + 1;

	while (left <= right) {
        mid = left + (right - left) / 2;
        if (entries[mid].key > key){
            right = mid - 1;
        }else if (entries[mid].key < key){
            left = mid + 1;
		}else{
            return mid; 
		}
    }

    if (left < n) {
        return left;
    }
    return -1;

}

void SortLnode(entry entry[], int start, int end, int pos[])
{
    if (start >= end) return;

    int pos_start = pos[start];
    unsigned long key = entry[pos_start].key;  
    int l, r;

    l = start;  r = end;
    while (l < r) {
        while ((l < r) && (entry[pos[r]].key > key)) r--;
        if (l < r) {
            pos[l] = pos[r];
            l++;
        }
        while ((l < r) && (entry[pos[l]].key <= key)) l++;
        if (l < r) {
            pos[r] = pos[l];
            r--;
        }
    }
    pos[l] = pos_start;
    SortLnode(entry, start, l - 1, pos);
    SortLnode(entry, l + 1, end, pos);
}

void QuickSort(entry *entries, int low, int high)
{
	int i = low;	
	int j = high;
	unsigned long key = entries[i].key;
	void *ptr = entries[i].ptr;
 
	while (i < j){					
		while(i < j && entries[j].key >= key){
			j--;
		}
		entries[i].key = entries[j].key;
		entries[i].ptr = entries[j].ptr;
 
		while(i < j && entries[i].key <= key){
			i++;
		}
		entries[j].key = entries[i].key;
		entries[j].ptr = entries[i].ptr;
	}
	entries[i].key = key;
	entries[i].ptr = ptr;
	if (i-1 > low) {
		QuickSort(entries, low, i-1);
	}
	if (i+1 < high){
		QuickSort(entries, i+1, high);
	}
	return;
}

void SelectSort(entry entries[] , int n)
{
	int i,j;
	int min = 0;
	for(i = 0; i < n - 1; i++){
		min = i;
		for(j = i + 1; j < n; j++){
			if(entries[min].key > entries[j].key){
				min = j;
			}
		}
		if (min != i){
			entry temp = entries[min];
			entries[min] = entries[i];
			entries[i] = temp;
		}else{
			continue;
		}
	}
}

void *Lookup_b(tree *t, unsigned long key){
	innerNode *iNode;
	leafNode *node;
restart:
	node = FindLeafNode_b(t, key);
    int i, sort, bitmap, lockLocation, index, left = 0, right, mid, version; 
	version = getlock(&node->hdr.version);
	sort = node->hdr.sort;
	if(sort == 3){
		bitmap = node->hdr.bitmap;
        for (i = 0; i < 3; i++)
        {
            index = WRITER_BUFFER_INDEX + i;
            if (bitmap & (1UL << i) && node->entries[index].key == key)
            {
                if (version != getlock(&node->hdr.version))
                {
                    goto restart;
                }
                return node->entries[index].ptr;
            }
        }
        index = node->hdr.left_index;
        lockLocation = node->left_lock;
		right = index - 1;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (node->entries[mid].key > key){
                right = mid - 1;
            }else if (node->entries[mid].key < key){
                left = mid + 1;
            }else{
				if(mid == 0 && node->hdr.min == 1){
					return NULL;
				}
                if (mid >= lockLocation)
                {
                    goto restart;
                }
                return node->entries[mid].ptr;
            }
        }
        right = node->hdr.right_index;
        lockLocation = node->right_lock;
		for (i = index; i < right; i++) {
            if (node->entries[i].key == key){
                if (i >= lockLocation)
                {
                    goto restart;
                }
                return node->entries[i].ptr;
            }
		}
	}else if (sort == 0){
		bitmap = node->hdr.bitmap;
        for (i = 0; i < 3; i++)
        {
            index = WRITER_BUFFER_INDEX + i;
            if (bitmap & (1UL << i) && node->entries[index].key == key)
            {
                if (version != getlock(&node->hdr.version))
                {
                    goto restart;
                }
                return node->entries[index].ptr;
            }
        }
        left = node->hdr.left_index;
        lockLocation = node->left_lock;
		for (i = 0; i < left; i++) {
			if (node->entries[i].key == key){
				if(i == 0 && node->hdr.min == 1){
					return NULL;
				}
				if (i >= lockLocation)
				{
					goto restart;
				}
				return node->entries[i].ptr;
			}
		}
        lockLocation = node->right_lock;
        right = node->hdr.right_index;
        index = node->hdr.sort_index;
		for (i = index; i < right; i++) {
            if (node->entries[i].key == key){
                if (i >= lockLocation)
                {
                    goto restart;
                }
                return node->entries[i].ptr;
            }
		}
	}else{
        index = node->hdr.left_index;
		right = index - 1;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (node->entries[mid].key > key){
                right = mid - 1;
            }else if (node->entries[mid].key < key){
                left = mid + 1;
            }else{
				if(mid == 0 && node->hdr.min == 1){
					return NULL;
				}
                if (version != getlock(&node->hdr.version))
                {
                    goto restart;
                }
                return node->entries[mid].ptr;
            }
        }
        left = index;
        right = NODE_SIZE - 1;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (node->entries[mid].key > key){
                right = mid - 1;
            }else if (node->entries[mid].key < key){
                left = mid + 1;
            }else{
                if (version != getlock(&node->hdr.version))
                {
                    goto restart;
                }
                return node->entries[mid].ptr;
            }
        }
	}
	return NULL;
}

void *Lookup_t(tree *t, unsigned long key){
	innerNode *iNode;
	leafNode *node;
restart:
	node = FindLeafNode_t(t, key);
    int i, sort, bitmap, lockLocation, index, left = 0, right, mid, version; 
	version = getlock(&node->hdr.version);
	sort = node->hdr.sort;
	if(sort == 3){
		bitmap = node->hdr.bitmap;
        for (i = 0; i < 3; i++)
        {
            index = WRITER_BUFFER_INDEX + i;
            if (bitmap & (1UL << i) && node->entries[index].key == key)
            {
                if (version != getlock(&node->hdr.version))
                {
                    goto restart;
                }
                return node->entries[index].ptr;
            }
        }
        index = node->hdr.left_index;
        lockLocation = node->left_lock;
		for (i = 0; i < index; i++) {
			if (node->entries[i].key <= key){
				if (node->entries[i].key == key){
					if(i == 0 && node->hdr.min == 1){
						return NULL;
					}
					if (i >= lockLocation)
					{
						goto restart;
					}
					return node->entries[i].ptr;
				}
			}else{
				break;
			}
		}
        right = node->hdr.right_index;
        lockLocation = node->right_lock;
		for (i = index; i < right; i++) {
            if (node->entries[i].key == key){
                if (i >= lockLocation)
                {
                    goto restart;
                }
                return node->entries[i].ptr;
            }
		}
	}else if (sort == 0){
		bitmap = node->hdr.bitmap;
        for (i = 0; i < 3; i++)
        {
            index = WRITER_BUFFER_INDEX + i;
            if (bitmap & (1UL << i) && node->entries[index].key == key)
            {
                if (version != getlock(&node->hdr.version))
                {
                    goto restart;
                }
                return node->entries[index].ptr;
            }
        }
        left = node->hdr.left_index;
        lockLocation = node->left_lock;
		for (i = 0; i < left; i++) {
			if (node->entries[i].key == key){
				if(i == 0 && node->hdr.min == 1){
					return NULL;
				}
				if (i >= lockLocation)
				{
					goto restart;
				}
				return node->entries[i].ptr;
			}
		}
        lockLocation = node->right_lock;
        right = node->hdr.right_index;
        index = node->hdr.sort_index;
		for (i = index; i < right; i++) {
            if (node->entries[i].key == key){
                if (i >= lockLocation)
                {
                    goto restart;
                }
                return node->entries[i].ptr;
            }
		}
	}else{
        index = node->hdr.left_index;
		for (i = 0; i < index; i++) {
			if (node->entries[i].key <= key){
				if (node->entries[i].key == key){
					if(i == 0 && node->hdr.min == 1){
						return NULL;
					}
					if (version != getlock(&node->hdr.version))
					{
						goto restart;
					}
					return node->entries[i].ptr;
				}
			}else{
				break;
			}
		}
		right = NODE_SIZE;
		for (i = index; i < right; i++) {
			if (node->entries[i].key <= key){
				if (node->entries[i].key == key){
					if (version != getlock(&node->hdr.version))
					{
						goto restart;
					}
					return node->entries[i].ptr;
				}
			}else{
				break;
			}
		}
	}
	return NULL;
}

void RangeLookUp(tree *t, unsigned long startKey, unsigned long endKey, bool methord){
	int i, version, count = 0, num, searchCount = 0, sort, sort_index, index, left, right;
	unsigned short bitmap;
restart:
	leafNode *startNode, *endNode, *nowNode;
	if(methord){
		startNode = FindLeafNode_b(t, startKey);
		endNode = FindLeafNode_b(t, endKey);
	}else{
		startNode = FindLeafNode_t(t, startKey);
		endNode = FindLeafNode_t(t, endKey);
	}
    static thread_local std::array <uint64_t, 100> buf;
	memset(buf.data(), 0, sizeof(buf));
	if (startNode != endNode){
startRestart1:
		sort = startNode->hdr.sort;
		if (lockstatus(&startNode->hdr.version))
		{
			goto restart;
		}
		version = getlock(&startNode->hdr.version);
		if(sort == 3){
			sort_index = startNode->hdr.sort_index;
			index = startNode->hdr.right_index;
			if(startNode->entries[0].key >= startKey && startNode->hdr.min != 1){
				buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[0].ptr);
			}
            for (i = 1; i < sort_index; i++){
                if (startNode->entries[i].key >= startKey){
					for(;i < sort_index; i++){
						buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
					}
					break;
				}	
            }
			for (i = sort_index; i < index; i++){
                if (startNode->entries[i].key >= startKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
                }				
            }
            bitmap = startNode->hdr.bitmap;
            for (i = 0; i < 3; i++){
                num = WRITER_BUFFER_INDEX + i;
                if ((bitmap & 1UL << i) && startNode->entries[num].key >= startKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[num].ptr);
                }
            }
		}else if (sort == 0){
			left = startNode->hdr.left_index;
			if(startNode->entries[0].key >= startKey && startNode->hdr.min != 1){
				buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[0].ptr);
			}
			for (i = 1; i < left; i++){
				if (startNode->entries[i].key >= startKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
				}
			}
			right = startNode->hdr.right_index;
            for (i = startNode->hdr.sort_index; i < right; i++){
                if (startNode->entries[i].key >= startKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
                }
            }
            bitmap = startNode->hdr.bitmap;
            for (i = 0; i < 3; i++){
                num = WRITER_BUFFER_INDEX + i;
                if ((bitmap & 1UL << i) && startNode->entries[num].key >= startKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[num].ptr);
                }
            }
		}else{
			index = startNode->hdr.sort_index;
            for (i = 0; i < index; i++){
				if (startNode->entries[i].key >= startKey){
					for(; i < index; i++){
						buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
					}
					break;
				}
			}
			for (i = index; i < NODE_SIZE; i++){
				if (startNode->entries[i].key >= startKey){
					for(; i < NODE_SIZE; i++){
						buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
					}
					break;
				}
			}
		}
		count = searchCount;
		if (version != getlock(&startNode->hdr.version))
		{
            searchCount = 0;
			goto startRestart1;
		}

		nowNode = startNode->nextPtr;
		if(nowNode->entries[0].key < startKey){
			startNode = startNode->nextPtr;
			searchCount = 0;
			goto startRestart1;
		}
nextRestart1:
		while (nowNode != endNode){
			count = searchCount;
			version = getlock(&nowNode->hdr.version);
			sort = nowNode->hdr.sort;
			if(sort == 3){
				index = nowNode->hdr.right_index;
				if(nowNode->hdr.min != 1){
					buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[0].ptr);
				}
				for (i = 1; i < index; i++){
					buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[i].ptr);
				}
                bitmap = nowNode->hdr.bitmap;
                for (i = 0; i < 3; i++){
                    if (bitmap & 1UL << i){
                        buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[WRITER_BUFFER_INDEX + i].ptr);
                    }
                }	
			}else if (sort == 0){
				left = nowNode->hdr.left_index;
				if(nowNode->hdr.min != 1){
					buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[0].ptr);
				}
                for (i = 1; i < left; i++){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[i].ptr);
                }
				right = nowNode->hdr.right_index;
                for (i = nowNode->hdr.sort_index; i < right; i++){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[i].ptr);
                }
                bitmap = nowNode->hdr.bitmap;
                for (i = 0; i < 3; i++){
                    if (bitmap & 1UL << i){
                        buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[WRITER_BUFFER_INDEX + i].ptr);
                    }
                }
			}else{
				for (i = 0; i < NODE_SIZE; i++){
					buf[searchCount++] = reinterpret_cast<uint64_t>(nowNode->entries[i].ptr);
				}
			}
			if (version != getlock(&nowNode->hdr.version))
			{
                searchCount = count;
				goto nextRestart1;
			}
			nowNode = nowNode->nextPtr;
		}

endRestart1:
		sort = endNode->hdr.sort;
		count = searchCount;
		version = getlock(&endNode->hdr.version);
		if (lockstatus(&endNode->hdr.version))
        {
            goto endRestart1;
        }
		if(sort == 3){
			index = endNode->hdr.right_index;
			sort_index = endNode->hdr.sort_index;
			if(endNode->entries[0].key <= endKey && endNode->hdr.min != 1){
				buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[0].ptr);
			}
            for (i = 1; i < sort_index; i++){
                if (endNode->entries[i].key > endKey){
					break;
                }
				buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[i].ptr);
            }
			for (i = sort_index; i < index; i++){
                if (endNode->entries[i].key <= endKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[i].ptr);
                }
            }
            bitmap = endNode->hdr.bitmap;
            for (i = 0; i < 3; i++){
                num = WRITER_BUFFER_INDEX + i;
                if ((bitmap & 1UL << i) && endNode->entries[num].key <= endKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[num].ptr);
                }
            }
		}else if (sort == 0){
			left = endNode->hdr.left_index;
			if(endNode->entries[0].key <= endKey && endNode->hdr.min != 1){
				buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[0].ptr);
			}
			for (i = 1; i < left; i++){
				if (endNode->entries[i].key <= endKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[i].ptr);
				}
			}
			right = endNode->hdr.right_index;
            for (i = endNode->hdr.sort_index; i < right; i++){
				if (endNode->entries[i].key <= endKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[i].ptr);
				}
			}
            bitmap = endNode->hdr.bitmap;
            for (i = 0; i < 3; i++){
                num = WRITER_BUFFER_INDEX + i;
                if ((bitmap & 1UL << i) && endNode->entries[num].key <= endKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[num].ptr);
                }
            }
		}else{
			sort_index = endNode->hdr.sort_index;
            for (i = 0; i < sort_index; i++){
				if (endNode->entries[i].key > endKey){
					break;
				}
				buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[i].ptr);
			}
			for (i = sort_index; i < NODE_SIZE; i++){
				if (endNode->entries[i].key > endKey){
					break;
				}
				buf[searchCount++] = reinterpret_cast<uint64_t>(endNode->entries[i].ptr);
			}
		}
		if (version != getlock(&endNode->hdr.version))
		{
            searchCount = count;
			goto endRestart1;
		}
		if(endNode->nextPtr && endNode->nextPtr != NULL){
			if(endKey > endNode->nextPtr->entries[0].key){
				endNode = endNode->nextPtr;
				goto endRestart1;
			}
		}
	}else{
startRestart2:
		count = searchCount;
		version = getlock(&startNode->hdr.version);
		sort = startNode->hdr.sort;
		if(sort == 3){
      		index = startNode->hdr.sort_index;
			if(startNode->entries[0].key >= startKey && startNode->entries[0].key <= endKey && startNode->hdr.min != 1){
				buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[0].ptr);
			}
			for (i = 1; i < index; i++){
				if (startNode->entries[i].key >= startKey){
					for(;i < index; i++){
						if(startNode->entries[i].key > endKey){
							break;
						}
						buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
					}
					break;
				}
			}
			right = startNode->hdr.right_index;
            for (i = index; i < right; i++){
				if (startNode->entries[i].key >= startKey && startNode->entries[i].key <= endKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
				}
			}
            bitmap = startNode->hdr.bitmap;
			for (i = 0; i < 3; i++){
                num = WRITER_BUFFER_INDEX + i;
				if ((bitmap & 1UL << i) && startNode->entries[num].key >= startKey && startNode->entries[num].key <= endKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[num].ptr);
				}
			}	
		}else if(sort == 0){
			left = startNode->hdr.left_index;
			if(startNode->entries[0].key >= startKey && startNode->entries[0].key <= endKey && startNode->hdr.min != 1){
				buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[0].ptr);
			}
			for (i = 1; i < left; i++){
				if (startNode->entries[i].key >= startKey && startNode->entries[i].key <= endKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
				}
			}
			right = startNode->hdr.right_index;
            for (i = startNode->hdr.sort_index; i < right; i++){
				if (startNode->entries[i].key >= startKey && startNode->entries[i].key <= endKey){
					buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
				}
			}
            bitmap = startNode->hdr.bitmap;
            for (i = 0; i < 3; i++){
                num = WRITER_BUFFER_INDEX + i;
                if ((bitmap & 1UL << i) && startNode->entries[num].key >= startKey && startNode->entries[num].key <= endKey){
                    buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[num].ptr);
                }
            }
		}else{
			index = startNode->hdr.sort_index;
			for (i = 0; i < index; i++){
				if (startNode->entries[i].key >= startKey){
					for(;i < index; i++){
						if(startNode->entries[i].key > endKey){
							break;
						}
						buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
					}
					break;
                }
			}
			for (i = index; i < NODE_SIZE; i++){
				if (startNode->entries[i].key >= startKey){
					for(;i < NODE_SIZE; i++){
						if(startNode->entries[i].key > endKey){
							break;
						}
						buf[searchCount++] = reinterpret_cast<uint64_t>(startNode->entries[i].ptr);
					}
					break;
                }
			}
		}
		if (version != getlock(&startNode->hdr.version))
		{
            searchCount = count;
			goto startRestart2;
		}
		if(startNode->nextPtr != NULL){
			if(endKey > startNode->nextPtr->entries[0].key){
				startNode = startNode->nextPtr;
				goto startRestart2;
			}
		}
	}
}

void Insert_b(tree *t, unsigned long key, void *value, LOG *mylog){
	leafNode *node;
restart:
	node = FindLeafNode_b(t, key);
    if (!vlock(&node->hdr.version)) { goto restart; }
    if(node->nextPtr && node->nextPtr != NULL){
        if(key > node->nextPtr->entries[0].key){
            vunlock(&node->hdr.version);
            goto restart;
        }
    }
    int i, bitmap, sort, index, left = 0, right, mid;
	sort = node->hdr.sort;

	if(bloom_maybe_contains(&node->hdr.filter, key))
	{
		if(sort == 3){
			bitmap = node->hdr.bitmap;
			for (i = 0; i < 3; i++)
			{
				if (bitmap & (1UL << i) && node->entries[WRITER_BUFFER_INDEX + i].key == key)
				{
					vunlock(&node->hdr.version);
					return;
				}
			}
			index = node->hdr.left_index;
			right = index - 1;
			while (left <= right) {
				mid = left + (right - left) / 2;
				if (node->entries[mid].key > key){
					right = mid - 1;
				}else if (node->entries[mid].key < key){
					left = mid + 1;
				}else{
					vunlock(&node->hdr.version);
					return;
				}
			}
			right = node->hdr.right_index;
			for (i = index; i < right; i++) {
				if (node->entries[i].key == key){
					vunlock(&node->hdr.version);
					return;
				}
			}
		}else if (sort == 0){
			bitmap = node->hdr.bitmap;
			for (i = 0; i < 3; i++)
			{
				if (bitmap & (1UL << i) && node->entries[WRITER_BUFFER_INDEX + i].key == key)
				{
					vunlock(&node->hdr.version);
					return;
				}
			}
			left = node->hdr.left_index;
			for (i = 0; i < left; i++) {
				if (node->entries[i].key == key){
					vunlock(&node->hdr.version);
					return;
				}
			}
			right = node->hdr.right_index;
			index = node->hdr.sort_index;
			for (i = index; i < right; i++) {
				if (node->entries[i].key == key){
					vunlock(&node->hdr.version);
					return;
				}
			}
		}else{
			index = node->hdr.left_index;
			right = index - 1;
			while (left <= right) {
				mid = left + (right - left) / 2;
				if (node->entries[mid].key > key){
					right = mid - 1;
				}else if (node->entries[mid].key < key){
					left = mid + 1;
				}else{
					vunlock(&node->hdr.version);
					return;
				}
			}
			left = node->hdr.sort_index;
			right = NODE_SIZE - 1;
			while (left <= right) {
				mid = left + (right - left) / 2;
				if (node->entries[mid].key > key){
					right = mid - 1;
				}else if (node->entries[mid].key < key){
					left = mid + 1;
				}else{
					vunlock(&node->hdr.version);
					return;
				}
			}
		}
	}

	if (sort != 4)
	{
		DirectInsertLeaf(t, node, key, value, mylog);
        vunlock(&node->hdr.version);
        return;
	}
    else
    {
        leafNode* splitNode = SplitLeaf(node);
        if (key < splitNode->entries[0].key){
            DirectInsertLeaf(t, node, key, value, mylog);
		}else{
			DirectInsertLeaf(t, splitNode, key, value, mylog);
		}
        AddInode(t, node, splitNode->entries[0].key, splitNode, t->height, true);
        vunlock(&node->hdr.version);
		vunlock(&splitNode->hdr.version);
        return;
    } 
}

void Insert_t(tree *t, uint64_t ii, unsigned long key, void *value, LOG *mylog){
	leafNode *node;
restart:
	node = FindLeafNode_t(t, key);
    if (!vlock(&node->hdr.version)) { goto restart; }
    if(node->nextPtr && node->nextPtr != NULL){
        if(key > node->nextPtr->entries[0].key){
            vunlock(&node->hdr.version);
            goto restart;
        }
    }
    int i, bitmap, sort, index, left = 0, right, mid;
	sort = node->hdr.sort;

	if(bloom_maybe_contains(&node->hdr.filter, key))
	{
		if(sort == 3){
			bitmap = node->hdr.bitmap;
			for (i = 0; i < 3; i++)
			{
				if (bitmap & (1UL << i) && node->entries[WRITER_BUFFER_INDEX + i].key == key)
				{
					vunlock(&node->hdr.version);
					return;
				}
			}
			index = node->hdr.left_index;
			for (i = 0; i < index; i++) {
				if (node->entries[i].key <= key){
					if (node->entries[i].key == key){
						vunlock(&node->hdr.version);
						return;
					}
				}else{
					break;
				}
			}
			right = node->hdr.right_index;
			for (i = index; i < right; i++) {
				if (node->entries[i].key == key){
					vunlock(&node->hdr.version);
					return;
				}
			}
		}else if (sort == 0){
			bitmap = node->hdr.bitmap;
			for (i = 0; i < 3; i++)
			{
				if (bitmap & (1UL << i) && node->entries[WRITER_BUFFER_INDEX + i].key == key)
				{
					vunlock(&node->hdr.version);
					return;
				}
			}
			left = node->hdr.left_index;
			for (i = 0; i < left; i++) {
				if (node->entries[i].key == key){
					vunlock(&node->hdr.version);
					return;
				}
			}
			right = node->hdr.right_index;
			index = node->hdr.sort_index;
			for (i = index; i < right; i++) {
				if (node->entries[i].key == key){
					vunlock(&node->hdr.version);
					return;
				}
			}
		}else{
			index = node->hdr.left_index;
			for (i = 0; i < index; i++) {
				if (node->entries[i].key <= key){
					if (node->entries[i].key == key){
						vunlock(&node->hdr.version);
						return;
					}
				}else{
					break;
				}
			}
			right = NODE_SIZE;
			for (i = index; i < right; i++) {
				if (node->entries[i].key <= key){
					if (node->entries[i].key == key){
						vunlock(&node->hdr.version);
 						return;
					}
				}else{
					break;
				}
			}
		}
	}

	if (sort != 4)
	{
		DirectInsertLeaf(t, node, key, value, mylog);
        vunlock(&node->hdr.version);
        return;
	}
    else
    {
        leafNode* splitNode = SplitLeaf(node);
        if (key < splitNode->entries[0].key){
            DirectInsertLeaf(t, node, key, value, mylog);
		}else{
			DirectInsertLeaf(t, splitNode, key, value, mylog);
		}
        AddInode(t, node, splitNode->entries[0].key, splitNode, t->height, false);
        vunlock(&node->hdr.version);
		vunlock(&splitNode->hdr.version);
        return;
    } 
}

void DirectInsertLeaf(tree *t, leafNode *node, unsigned long key, void *value, LOG *mylog){
	int i, s, ss, insert_num;
    int lazynum = lazyfield_count(node->hdr.bitmap);
    if (lazynum < 3){
        for (ss = 0; ss < 3; ss++)
        {
            if (comparebitmap(node->hdr.bitmap, ss) == 0)
            {
                insert_num = WRITER_BUFFER_INDEX + ss;
                node->entries[insert_num].key = key;
                node->entries[insert_num].ptr = value;
                node->hdr.bitmap |= 1UL << ss;
				bloom_insert(&node->hdr.filter, key);
                if (node->hdr.right_index != WRITER_BUFFER_INDEX || ss != 2)
                {
                    clflush(&node->entries[WRITER_BUFFER_INDEX], CACHE_LINE_SIZE);
                    return ;
                }else{
                    SortRight(t, node, mylog);
                    node->right_lock = NODE_SIZE;
                    return ;
                }
            }
        }
    }else if(node->hdr.sort == 0){
        insert_num = node->hdr.left_index;
        if(insert_num < node->hdr.sort_index - 3){
            int entry_num = insert_num;
            for (i = WRITER_BUFFER_INDEX; i < NODE_SIZE; i++)
            {
                node->entries[entry_num].key = node->entries[i].key;
                node->entries[entry_num].ptr = node->entries[i].ptr;
                entry_num++;
            }
            node->entries[entry_num].key = key;
            node->entries[entry_num].ptr = value;
            entry_num++;
            clflush(&node->entries[insert_num], CACHE_LINE_SIZE);	
            node->hdr.bitmap = 0;  
            node->hdr.left_index = entry_num;
			node->left_lock = entry_num;
			bloom_insert(&node->hdr.filter, key);
            clflush(&node->hdr, CACHE_LINE_SIZE);
            if(entry_num == node->hdr.sort_index){
                SortLeft(t, node, mylog);
            }
            return;
        } else {
            node->entries[insert_num].key = key;
            node->entries[insert_num].ptr = value;
            clflush(&node->entries[insert_num], CACHE_LINE_SIZE);
            insert_num++;
            node->hdr.left_index = insert_num;
			node->left_lock = insert_num;
			bloom_insert(&node->hdr.filter, key);
            clflush(&node->hdr, CACHE_LINE_SIZE);
            if(insert_num == node->hdr.sort_index){
                SortLeft(t, node, mylog);
            }
            return;
        }
    }else{
        insert_num = node->hdr.right_index;
        if(insert_num < WRITER_BUFFER_INDEX - 3){
            int entry_num = insert_num;
            for (i = WRITER_BUFFER_INDEX; i < NODE_SIZE; i++)
            {
                node->entries[entry_num].key = node->entries[i].key;
                node->entries[entry_num].ptr = node->entries[i].ptr;
                entry_num++;
            }
            node->entries[entry_num].key = key;
            node->entries[entry_num].ptr = value;
            entry_num++;
            clflush(&node->entries[insert_num], CACHE_LINE_SIZE);	
            node->hdr.bitmap = 0;  
            node->hdr.right_index = entry_num;
			node->right_lock = entry_num;
			bloom_insert(&node->hdr.filter, key);
            clflush(&node->hdr, CACHE_LINE_SIZE);
            return;
        } else {
            node->entries[insert_num].key = key;
            node->entries[insert_num].ptr = value;
            clflush(&node->entries[insert_num], CACHE_LINE_SIZE);
            insert_num++;
            node->hdr.right_index = insert_num;
			node->right_lock = insert_num;
			bloom_insert(&node->hdr.filter, key);
            clflush(&node->hdr, CACHE_LINE_SIZE);
            if(insert_num == WRITER_BUFFER_INDEX){
                SortRight(t, node, mylog);
            }
            return;
        }
    }
}

void SortLeft(tree *t, leafNode *node, LOG *mylog){
    int i, left_index = node->hdr.left_index;
leftrestart:
	add_entry_to_log(mylog, node->entries, 0, left_index);
    entry sorted_entry[left_index];
    for (i = 0; i < left_index; i++){
        sorted_entry[i].key = node->entries[i].key;
        sorted_entry[i].ptr = node->entries[i].ptr;
    }
    QuickSort(sorted_entry, 0, left_index - 1);
	node->left_lock = 0;
    for (i = 0; i < left_index; i++){
        node->entries[i].key = sorted_entry[i].key;
        node->entries[i].ptr = sorted_entry[i].ptr;
    }
	node->left_lock = left_index;
    if (node->hdr.right_index == WRITER_BUFFER_INDEX && node->hdr.bitmap == 7){
        node->hdr.sort = 4;
    }else{
        node->hdr.sort = 3;
    }

    delclflushmore(node, &node->entries[left_index - 1]);
    clflush(&node->hdr, CACHE_LINE_SIZE);
}

void SortRight(tree *t, leafNode *node, LOG *mylog){
    int i, j, num, sort_index;
rightrestart:
    sort_index = node->hdr.sort_index;
    num = NODE_SIZE - sort_index;
	add_entry_to_log(mylog, node->entries, sort_index, num);
    entry sorted_entry[num];
    for (i = sort_index, j = 0; i < NODE_SIZE; i++, j++){
        sorted_entry[j].key = node->entries[i].key;
        sorted_entry[j].ptr = node->entries[i].ptr;
    }
    QuickSort(sorted_entry, 0, num - 1);
	node->right_lock = 0;
    for (i = sort_index, j = 0; i < NODE_SIZE; i++, j++){
        node->entries[i].key = sorted_entry[j].key;
        node->entries[i].ptr = sorted_entry[j].ptr;
    }
	node->right_lock = NODE_SIZE;
    if (node->hdr.left_index == sort_index){
        node->hdr.sort = 4;
    }else{
        node->hdr.sort = 0;
    }
    delclflushmore(&node->entries[sort_index], &node->hdr);
}

leafNode *SplitLeaf(leafNode *node){
	int curLeft, curRight, appendLocation = 0, sort_index = node->hdr.sort_index;
	unsigned long leftMid = node->entries[sort_index / 2].key, rightMid = node->entries[(NODE_SIZE - sort_index) / 2 + sort_index].key, mid;
	uint64_t bitmap = 7, new_bitmap = 0;
	mid = (leftMid + rightMid) / 2;
	curLeft = GetMidLocation(node->entries, mid, 0, sort_index - 1);
	curRight = GetMidLocation(node->entries, mid, sort_index, NODE_SIZE - 1);
	leafNode *splitNode = allocLeafNode();
	vlock(&splitNode->hdr.version);
	if(curLeft != -1 && curRight != -1){
        node->left_lock = curLeft;
        if(curRight >= WRITER_BUFFER_INDEX){
            node->right_lock = WRITER_BUFFER_INDEX;
        }else{
            node->right_lock = curRight;
        }
		while ((curLeft < sort_index) && (curRight < NODE_SIZE)){
			if (node->entries[curLeft].key < node->entries[curRight].key){
                splitNode->entries[appendLocation].key = node->entries[curLeft].key;
                splitNode->entries[appendLocation].ptr = node->entries[curLeft].ptr;
				bloom_insert(&splitNode->hdr.filter,splitNode->entries[appendLocation].key);
				appendLocation++;
				curLeft++;
			}else{
                splitNode->entries[appendLocation].key = node->entries[curRight].key;
                splitNode->entries[appendLocation].ptr = node->entries[curRight].ptr;
                bloom_insert(&splitNode->hdr.filter,splitNode->entries[appendLocation].key);
				if(curRight >= WRITER_BUFFER_INDEX){
					bitmap &= ~(1UL<<(curRight % WRITER_BUFFER_INDEX));
				}
				appendLocation++;
				curRight++;
			}
		}

		if (curLeft == sort_index){
			for (; curRight < NODE_SIZE; curRight++){
                splitNode->entries[appendLocation].key = node->entries[curRight].key;
                splitNode->entries[appendLocation].ptr = node->entries[curRight].ptr;
				bloom_insert(&splitNode->hdr.filter,splitNode->entries[appendLocation].key);
                if(curRight >= WRITER_BUFFER_INDEX){
					bitmap &= ~(1UL<<(curRight % WRITER_BUFFER_INDEX));
				}
				appendLocation++;
			}
		}else if (curRight == NODE_SIZE){
			for (; curLeft < sort_index; curLeft++){
                splitNode->entries[appendLocation].key = node->entries[curLeft].key;
                splitNode->entries[appendLocation].ptr = node->entries[curLeft].ptr;
				bloom_insert(&splitNode->hdr.filter,splitNode->entries[appendLocation].key);
				appendLocation++;
			}
		}
        splitNode->hdr.sort = 3;
        splitNode->hdr.left_index = appendLocation;
        splitNode->hdr.right_index = appendLocation;
        splitNode->hdr.sort_index = appendLocation;
		splitNode->left_lock = appendLocation;
        splitNode->nextPtr = node->nextPtr;
		delclflushmore(splitNode, &(splitNode->entries[appendLocation - 1]));
        clflush(&splitNode->hdr, CACHE_LINE_SIZE);
        node->hdr.left_index = node->left_lock;
        node->hdr.right_index = node->right_lock;
		node->hdr.bitmap = bitmap;
        node->hdr.sort = 0;
		node->nextPtr = splitNode;
		clflush(&node->hdr, CACHE_LINE_SIZE);
		return splitNode;
	}else if(curLeft == -1){
		node->left_lock = sort_index;
        if(curRight >= WRITER_BUFFER_INDEX){
            node->right_lock = WRITER_BUFFER_INDEX;
        }else{
            node->right_lock = curRight;
        }
		for (; curRight < NODE_SIZE; curRight++){
            splitNode->entries[appendLocation].key = node->entries[curRight].key;
            splitNode->entries[appendLocation].ptr = node->entries[curRight].ptr;
			bloom_insert(&splitNode->hdr.filter,splitNode->entries[appendLocation].key);
            if(curRight >= WRITER_BUFFER_INDEX){
				bitmap &= ~(1UL<<(curRight % WRITER_BUFFER_INDEX));
			}
			appendLocation++;
		}
        splitNode->hdr.sort = 3;
        splitNode->hdr.left_index = appendLocation;
        splitNode->hdr.right_index = appendLocation;
        splitNode->hdr.sort_index = appendLocation;
		splitNode->left_lock = appendLocation;
        splitNode->nextPtr = node->nextPtr;
		delclflushmore(splitNode,&(splitNode->entries[appendLocation - 1]));
        clflush(&splitNode->hdr, CACHE_LINE_SIZE);
		node->hdr.sort = 3;
        node->hdr.right_index = node->right_lock;
		node->hdr.bitmap = bitmap;
		node->nextPtr = splitNode;
        clflush(&node->hdr, CACHE_LINE_SIZE);
		return splitNode;
	}else if(curRight == -1){
		node->left_lock = curLeft; 
        node->right_lock = NODE_SIZE;
		for (; curLeft < sort_index; curLeft++){
            splitNode->entries[appendLocation].key = node->entries[curLeft].key;
            splitNode->entries[appendLocation].ptr = node->entries[curLeft].ptr;
			bloom_insert(&splitNode->hdr.filter,splitNode->entries[appendLocation].key);
			appendLocation++;
		}
        splitNode->hdr.sort = 3;
        splitNode->hdr.left_index = appendLocation;
        splitNode->hdr.right_index = appendLocation;
        splitNode->hdr.sort_index = appendLocation;
		splitNode->left_lock = appendLocation;
		splitNode->nextPtr = node->nextPtr;
		delclflushmore(splitNode,&(splitNode->entries[appendLocation - 1]));
        clflush(&splitNode->hdr, CACHE_LINE_SIZE);
        node->hdr.sort = 0;
        node->hdr.left_index = node->left_lock;
		node->nextPtr = splitNode;
		clflush(&node->hdr, CACHE_LINE_SIZE);
		return splitNode;
	}
	return NULL;
}

int GetMidLocation(entry entries[], unsigned long key, int left, int right){
    int result = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (entries[mid].key > key) {
            result = mid; 
            right = mid - 1;
        } else if (entries[mid].key < key){
			left = mid + 1; 
		}
		else{
            return mid;
        }
    }
    return result;
}

void AddInode(tree *t, void *currentNode, unsigned long key, void *splitNode, unsigned int height, bool methord)
{
    innerNode *iNode;
    int i, num, level;
restart:
    if (height == 1)
    {
        innerNode *newRoot = allocInnerNode();
		setNewRoot(newRoot, currentNode, key, splitNode);
		setRoot(t, newRoot);
		return ;
	}
	if(methord){
		iNode = FindInnerNode_b(t, key, height);
	}else{
		iNode = FindInnerNode_t(t, key, height);
	}
	if (!vlock(&iNode->version)) { 
		goto restart; 
	}
  	else{

		if (iNode->num < NODE_SIZE){
			int hk = iNode->num;    
			for (; hk > 0; hk--)
			{
				if (iNode->entries[hk - 1].key < key)
				{                    
					iNode->entries[hk].key = key;
					iNode->entries[hk].ptr = splitNode;
					iNode->num = iNode->num + 1;
					break;
				}
				else
				{
					iNode->entries[hk].key = iNode->entries[hk - 1].key;
					iNode->entries[hk].ptr = iNode->entries[hk - 1].ptr;                    
				}
			}
			if (hk == 0)
			{
				iNode->entries[0].key = key;
				iNode->entries[0].ptr = splitNode;
				iNode->num = iNode->num + 1;
				delclflushmore(iNode,&(iNode->entries[iNode->num - 1]));
				vunlock(&iNode->version);
				return;
			}
			delclflushmore(&(iNode->entries[hk]),&(iNode->entries[iNode->num - 1]));
			clflush(iNode, CACHE_LINE_SIZE);
			vunlock(&iNode->version);
			return;
		}
		else
		{
			InodeSplit(t, iNode, key, splitNode, height, methord);
			return;
		}
	}
}

void InodeSplit(tree *t, innerNode *iNode, unsigned long key, void *node, unsigned int height, bool methord)
{ 
	int newNodeNum = 0, i;
    innerNode *newNode = allocInnerNode();
	vlock(&newNode->version);
	newNode->nextPtr = iNode->nextPtr;

    for (i = MID_NODE_SIZE; i < NODE_SIZE; i++)
    {
        newNode->entries[newNodeNum].key = iNode->entries[i].key;
        newNode->entries[newNodeNum].ptr = iNode->entries[i].ptr;
        newNodeNum++;
    }

    if (key >= newNode->entries[0].key)
    {
        int s = newNodeNum;
        for (; s > 0; s--)
        {
            if (newNode->entries[s - 1].key < key)
            {
                newNode->entries[s].key = key;
                newNode->entries[s].ptr = node;
                break;
            }
            else
            {
                newNode->entries[s].key = newNode->entries[s - 1].key;
                newNode->entries[s].ptr = newNode->entries[s - 1].ptr;
                if (s == 1)
                {
                    newNode->entries[s - 1].key = key;
                    newNode->entries[s - 1].ptr = node;
                    break;
                }
            }
        }
        newNode->num = newNodeNum + 1;
		delclflushmore(newNode,&(newNode->entries[newNode->num - 1]));
		iNode->num = MID_NODE_SIZE;
		iNode->nextPtr = newNode;
		delclflushmore(iNode,&(iNode->entries[iNode->num - 1]));
    }
    else
    {
        newNode->num = newNodeNum;
        delclflushmore(newNode,&(newNode->entries[newNode->num - 1]));
        iNode->num = MID_NODE_SIZE;

        for (i = MID_NODE_SIZE; i > 0; i--)
        {
            if (iNode->entries[i - 1].key < key)
            {
                iNode->entries[i].key = key;
                iNode->entries[i].ptr = node;
                break;
            }
            else
            {
                iNode->entries[i].key = iNode->entries[i - 1].key;
                iNode->entries[i].ptr = iNode->entries[i - 1].ptr;
                if (i == 1)
                {
                    iNode->entries[i - 1].key = key;
                    iNode->entries[i - 1].ptr = node;
                    break;
                }
            }
        }
        iNode->num = iNode->num + 1;
		iNode->nextPtr = newNode;
		delclflushmore(iNode,&(iNode->entries[iNode->num - 1]));
    }

	AddInode(t, iNode, newNode->entries[0].key, newNode, height - 1, methord);
	vunlock(&iNode->version);
	vunlock(&newNode->version);
    return;
}

void Delete(tree *t, unsigned long key){
    leafNode *node;
restart:
	node = FindLeafNode_t(t, key);
	if (!vlock(&node->hdr.version)) { goto restart; }
    int i, left, right, sort, sort_index, bitmap;
	sort = node->hdr.sort, sort_index = node->hdr.sort_index, bitmap = node->hdr.bitmap;
	bool flushed_entry = false, flushed_hdr = false;
	for (i = 0; i < 3; i++)
	{
		if (bitmap & (1UL << i) && node->entries[WRITER_BUFFER_INDEX + i].key == key)
		{
			node->hdr.bitmap &= ~(1UL<<i);
			flushed_hdr = true;
			goto deleted;
		}
	}
	right = node->hdr.right_index;
	for (i = sort_index; i < right; i++) {
		if (node->entries[i].key == key) {
			node->right_lock = i;
			if (i != right - 1) {
				node->entries[i].key = node->entries[right - 1].key;
				node->entries[i].ptr = node->entries[right - 1].ptr;
				flushed_entry = true;
			}
			node->hdr.right_index--;
			node->right_lock = node->hdr.right_index;
			if (sort == 4) node->hdr.sort = 3;
			flushed_hdr = true;
			goto deleted;
		}
	}
    left = (sort == 0) ? node->hdr.left_index : sort_index;
    for (i = 0; i < left; i++) {
        if (node->entries[i].key == key) {
            if (i == 0 && !node->hdr.min) {
                node->hdr.min = 1;
                flushed_hdr = true;
                goto deleted;
            } else if (i == 0) {
                vunlock(&node->hdr.version);
				falsedelete++;
                return;
            }
			node->left_lock = i;
            if (i != left - 1) {
                node->entries[i].key = node->entries[left - 1].key;
				node->entries[i].ptr = node->entries[left - 1].ptr;
                flushed_entry = true;
            }
            node->hdr.left_index--;
			node->left_lock = node->hdr.left_index;
            if (sort != 0) node->hdr.sort = 0;
            flushed_hdr = true;
            goto deleted;
        }
    }
	vunlock(&node->hdr.version);
	return;

deleted:
	if (flushed_entry)
        clflush(&node->entries[i], CACHE_LINE_SIZE);
    if (flushed_hdr)
        clflush(&node->hdr, CACHE_LINE_SIZE);
	if(node->hdr.left_index == 1 && node->hdr.min == 1 && node->hdr.right_index == node->hdr.sort_index && node->hdr.bitmap == 0){
		node->hdr.deleted = 1;
		if(t->height > 1){
			DelINode(t, key, t->height - 1);
		}
		if (node->nextPtr != nullptr)
		{
			if (node->nextPtr->hdr.deleted)
			{
				if (vlock(&node->nextPtr->hdr.version))
				{
					if (t->root == node)
					{
						t->root = node->nextPtr;
						clflush(t, CACHE_LINE_SIZE);
						vunlock(&node->nextPtr->hdr.version);
						return;
					}
					else
					{
						node->nextPtr = node->nextPtr->nextPtr;
						clflush(&node->hdr, CACHE_LINE_SIZE);
						vunlock(&node->hdr.version);
						return;
					}
				}
			}
		}
		clflush(&node->hdr, CACHE_LINE_SIZE);
	}

    vunlock(&node->hdr.version);
}

void DelINode(tree *t, unsigned long key, int height)
{
    innerNode *iNode;
    int i, num, level, flag = 0;
restart:
    iNode = (innerNode *)t->root;
    level = height;
    for (; level > 1; level--)
    {
		int version = getlock(&iNode->version);
		if (lockstatus(&iNode->version))
        {
            goto restart;
        } 
		num = iNode->num;
		if(key < iNode->entries[0].key){
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->leftmostPtr;
			flag = 1;
			continue;
		}
        for(i = 1; i < num; i++){
			if(key < iNode->entries[i].key){
				if (version != getlock(&iNode->version))
                {
                    goto restart;
                }
				iNode = (innerNode *)iNode->entries[i - 1].ptr;
				break;
			}
		}
		if(i == num){
			if (version != getlock(&iNode->version))
			{
				goto restart;
			}
			iNode = (innerNode *)iNode->entries[i - 1].ptr;
		}
    }
    
    if (!vlock(&iNode->version)) { 
		goto restart; 
	}
    num = iNode->num;
    if (num > 0)
    {              
        if (num == 1 && height == 1) 
        {
            void *temp;
			if (key < iNode->entries[0].key)
            {
                temp = iNode->leftmostPtr;
            }
            else
            {
                temp = iNode->entries[0].ptr;
            }
            if (t->height == 2)
            {
                if (!vlock(&((leafNode *)temp)->hdr.version))
                {
					vunlock(&iNode->version);
                    goto restart;
                }
                t->root = temp;
                t->height--;
                clflush(t, CACHE_LINE_SIZE);
                vunlock(&iNode->version);
				vunlock(&((leafNode *)temp)->hdr.version);
                return;
            }
            else
            {
                if (!vlock(&(((innerNode *)temp)->version)))
                {
                    vunlock(&iNode->version);
                    goto restart;
                }
                t->root = temp;
                t->height--;
                clflush(t, CACHE_LINE_SIZE);
                vunlock(&iNode->version);
				vunlock(&(((innerNode *)temp)->version));
                return;
            }           
        }
        else
        {
            if (key < iNode->entries[1].key)
            {
		        if(key < iNode->entries[0].key){
               	   iNode->leftmostPtr = iNode->entries[0].ptr;
		        }
                for (int ms = 1; ms <= num; ms++)
                {
                    iNode->entries[ms-1].key = iNode->entries[ms].key;
                    iNode->entries[ms-1].ptr = iNode->entries[ms].ptr;
                }
                iNode->num = num - 1;                
                delclflushmore(iNode, &(iNode->entries[iNode->num].key));
                vunlock(&iNode->version);
                return;
            }
            else
            {
                if (key >= iNode->entries[num - 1].key)
                {
                    iNode->num = num - 1;   
                    clflush(iNode, CACHE_LINE_SIZE);
                    vunlock(&iNode->version);
                    return;
                }
                else
                {
                    int bg = 0;
                    int ed = num - 1;
                    int middl = 0;
                    while (bg <= ed)
                    {
                        middl = (bg + ed) / 2;
                        if (iNode->entries[middl].key <= key && iNode->entries[middl + 1].key > key)
                        {
                            for (int ms = middl; ms < num; ms++)
                            {
                                iNode->entries[ms].key = iNode->entries[ms + 1].key;
                                iNode->entries[ms].ptr = iNode->entries[ms + 1].ptr;
                            }
                            iNode->num = num - 1;  
                            delclflushmore(&(iNode->entries[middl].key), &(iNode->entries[iNode->num - 1].key));
                            clflush(iNode, CACHE_LINE_SIZE);
							vunlock(&iNode->version);
                            return;                            
                        }
                        else
                        {
                            if (iNode->entries[middl].key > key)
                            {
                                ed = middl - 1;
                            }
                            else
                            {
                                bg = middl + 1;
                            }
                        }
                    }
                }
            } 
        }               
    }
    else
    {
        DelINode(t, key, height - 1);
        vunlock(&iNode->version);
    }   
}

void printLeaf(leafNode *node){
	printf("node: sort = %d\n",node->hdr.sort);
    printf("left_index = %d, right_index = %d, sort_index: %d, bitmap = %x\n",node->hdr.left_index,node->hdr.right_index,node->hdr.sort_index,node->hdr.bitmap);
    int bitmap = node->hdr.bitmap,sort_index = node->hdr.sort_index,left_index = node->hdr.left_index,right_index = node->hdr.right_index;
    for (int i = 0; i < WRITER_BUFFER_INDEX; i++)
    {
        if(i < left_index){
            printf("%lld  ",node->entries[i].key);
        }else if( i >= left_index && i < sort_index){
            printf("0  ");
        }else if(i >= sort_index && i < right_index){
            printf("%lld  ",node->entries[i].key);
        }else{
            printf("0  ");
        }
    }
    for (int i = WRITER_BUFFER_INDEX,j = 0; i < NODE_SIZE; i++,j++)
    {
        if(bitmap & (1 << j)){
            printf("%ld  ",node->entries[i].key);
        }else{
            printf("0  ");
        }
    }
    printf("\n");
}

void printInnerNode(innerNode *iNode){
    for(int j = 0;j < iNode->num;j++){
            printf("%ld  ",iNode->entries[j].key);
    }
    printf("\n");
    printf("innode->version = %ld, inode->num = %d\n",iNode->version.load(),iNode->num);
}