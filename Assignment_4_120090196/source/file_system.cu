#include "file_system.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

// helper function declaration
__device__ size_t my_strlen(const char *str);
__device__ int my_strcmp(const char *str1, const char *str2);
__device__ char *my_strcpy(char *dst, const char *src);
__device__ int get_empty_block(FileSystem *fs, int n);
__device__ int cmpLS_D(FileSystem *fs, int i, int j);
__device__ int cmpLS_S(FileSystem *fs, int i, int j);
typedef int (*CmpFunc)(FileSystem *, int, int);
__device__ void bubble_sort(FileSystem *fs, int *array, int len, CmpFunc cmp);
__device__ int compute_nblk(FileSystem *fs, int file_sz);
__device__ void clear_super_blk(FileSystem *fs, int addr, int nblk);
__device__ int ffsl(uint64_t num);
__device__ int get_block_from_2_unit(FileSystem *fs, int n1, int n2, int nblk);
__device__ void compact(FileSystem *fs);
__device__ __managed__ u32 gtime = 0;
#define MAX_BLOCK_NUM_ 4096 * 8
__device__ int addr2fd[MAX_BLOCK_NUM_];

__device__ static uint64_t MASK[] = {
    0x0ULL,
    0x1ULL,
    0x3ULL,
    0x7ULL,
    0xfULL,
    0x1fULL,
    0x3fULL,
    0x7fULL,
    0xffULL,
    0x1ffULL,
    0x3ffULL,
    0x7ffULL,
    0xfffULL,
    0x1fffULL,
    0x3fffULL,
    0x7fffULL,
    0xffffULL,
    0x1ffffULL,
    0x3ffffULL,
    0x7ffffULL,
    0xfffffULL,
    0x1fffffULL,
    0x3fffffULL,
    0x7fffffULL,
    0xffffffULL,
    0x1ffffffULL,
    0x3ffffffULL,
    0x7ffffffULL,
    0xfffffffULL,
    0x1fffffffULL,
    0x3fffffffULL,
    0x7fffffffULL,
    0xffffffffULL,
    0x1ffffffffULL,
    0x3ffffffffULL,
    0x7ffffffffULL,
    0xfffffffffULL,
    0x1fffffffffULL,
    0x3fffffffffULL,
    0x7fffffffffULL,
    0xffffffffffULL,
    0x1ffffffffffULL,
    0x3ffffffffffULL,
    0x7ffffffffffULL,
    0xfffffffffffULL,
    0x1fffffffffffULL,
    0x3fffffffffffULL,
    0x7fffffffffffULL,
    0xffffffffffffULL,
    0x1ffffffffffffULL,
    0x3ffffffffffffULL,
    0x7ffffffffffffULL,
    0xfffffffffffffULL,
    0x1fffffffffffffULL,
    0x3fffffffffffffULL,
    0x7fffffffffffffULL,
    0xffffffffffffffULL,
    0x1ffffffffffffffULL,
    0x3ffffffffffffffULL,
    0x7ffffffffffffffULL,
    0xfffffffffffffffULL,
    0x1fffffffffffffffULL,
    0x3fffffffffffffffULL,
    0x7fffffffffffffffULL,
    0xffffffffffffffffULL,
};

/**
 * superblock: Volume control block (master file table) contains volume details
 * File Control Block (FCB): contains many details about the file
 */
// global structure of volume
// super block(4KB) / FCB(32KB) / contents of the files(1024KB)
// The maximum size of a file name is 20 bytes
#define MAX_FILE_NAME_LEN 20
#define INVALID UINT32_MAX

// the structure of File Control Block
typedef struct FCB {
  char file_name[MAX_FILE_NAME_LEN]; // The maximum size of a file name is 20
                                     // bytes.
  uint16_t m_time;                   // modify time
  uint16_t c_time;                   // create time
  uint32_t file_sz;                  // file size
  uint32_t addr;                     // location of the file contents
} FCB;

#define AddrToFCB(addr) ((FCB *)(addr))

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE,
                        int FILE_BASE_ADDRESS) {
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  memset(volume, 0, VOLUME_SIZE);
  for (int i = 0; i < MAX_BLOCK_NUM_; i++) {
    addr2fd[i] = -1;
  }
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op) {
  /* Implement open operation here */
  int first_empty_file_pos = -1;
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
    // Get the address of the i-th FCB
    FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
    // compare filename
    if (first_empty_file_pos == -1 && my_strlen(fcb->file_name) == 0) {
      first_empty_file_pos = i;
    }
    if (my_strcmp(s, fcb->file_name) == 0) {
      return i;
    }
  }
  // this file not exists
  if (op == G_WRITE) {
    if (first_empty_file_pos == -1) {
      // no empty file
      return INVALID;
    }
    // create a new zero byte file.
    FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE +
                         fs->FCB_SIZE * first_empty_file_pos);

    my_strcpy(fcb->file_name, s);
    fcb->file_sz = 0;
    fcb->c_time = gtime;
    fcb->m_time = gtime;
    fcb->addr = 0; // empty file no block
    gtime++;
    return first_empty_file_pos;
  } else if (op == G_READ) {
    return INVALID;
  } else {
    // should not run here
    return INVALID;
  }
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp) {
  /* Implement read operation here */
  if (fp == INVALID || fp >= fs->MAX_FILE_NUM) {
    return;
  }

  FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp);
  if (my_strlen(fcb->file_name) == 0 || size > fcb->file_sz) {
    return;
  }

  int storage_start_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->FCB_ENTRIES;
  uchar *to_read =
      &fs->volume[storage_start_addr + fcb->addr * fs->STORAGE_BLOCK_SIZE];
  for (int i = 0; i < size; i++) {
    output[i] = to_read[i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp) {
RETRY:
  /* Implement write operation here */
  if (fp == INVALID || fp >= fs->MAX_FILE_NUM) {
    return INVALID;
  }

  FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp);
  if (my_strlen(fcb->file_name) == 0) {
    return INVALID;
  }

  int old_nblk = compute_nblk(fs, fcb->file_sz);
  int new_nblk = compute_nblk(fs, size);
  if (old_nblk != new_nblk) {
    // need release old space and allocate new space
    clear_super_blk(fs, fcb->addr, old_nblk);
    int pos = get_empty_block(fs, new_nblk);
    // printf("fs_write addr %d\n", pos);
    if (pos == -1) {
      // maybe need compact
      // printf("need compact\n");
      compact(fs);
      goto RETRY;
    } else {
      fcb->addr = pos;
      addr2fd[fcb->addr] = fp;
    }
  }
  fcb->file_sz = size;
  fcb->m_time = gtime++;
  int storage_start_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->FCB_ENTRIES;
  uchar *to_write =
      &fs->volume[storage_start_addr + fcb->addr * fs->STORAGE_BLOCK_SIZE];
  for (int i = 0; i < size; i++) {
    to_write[i] = input[i];
  }

  return 0;
}

__device__ void fs_gsys(FileSystem *fs, int op) {
  /* Implement LS_D and LS_S operation here */
  int fcbs[1024];
  int num = 0;
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
    FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
    if (my_strlen(fcb->file_name) != 0) {
      fcbs[num++] = i;
    }
  }

  if (op == LS_D) {
    printf("===sort by modified time===\n");
    bubble_sort(fs, fcbs, num, cmpLS_D);
    for (int i = 0; i < num; i++) {
      FCB *fcb =
          AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fcbs[i]);
      printf("%s\n", fcb->file_name);
    }
  } else if (op == LS_S) {
    printf("===sort by file size===\n");
    bubble_sort(fs, fcbs, num, cmpLS_S);
    for (int i = 0; i < num; i++) {
      FCB *fcb =
          AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fcbs[i]);
      printf("%s %d\n", fcb->file_name, fcb->file_sz);
    }
  } else {
    // should not run here
    return;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s) {
  /* Implement rm operation here */
  if (op != RM)
    return;
  for (int i = 0; i < fs->FCB_ENTRIES; i++) {
    // Get the address of the i-th FCB
    FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
    // compare filename
    if (my_strcmp(s, fcb->file_name) == 0) {
      // compute the blocks it occupies
      int nblk = compute_nblk(fs, fcb->file_sz);
      // clear super block(bitmap)
      clear_super_blk(fs, fcb->addr, nblk);
      // clear fcb, I judge by whether the filename is empty
      fcb->file_name[0] = 0;
    }
  }
}

/****************************** helper function ******************************/
__device__ size_t my_strlen(const char *str) {
  const char *eos = str;
  while (*eos++)
    ;
  return (eos - str - 1);
}

__device__ int my_strcmp(const char *str1, const char *str2) {
  int ret = 0;
  while (!(ret = *(unsigned char *)str1 - *(unsigned char *)str2) && *str1) {
    str1++;
    str2++;
  }

  if (ret < 0) {
    return -1;
  } else if (ret > 0) {
    return 1;
  }
  return 0;
}

__device__ char *my_strcpy(char *dst, const char *src) {
  char *ret = dst;
  while ((*dst++ = *src++) != '\0')
    ;
  return ret;
}

__device__ int end_consecutive_nblk(FileSystem *fs, int n) {
  uint64_t *p = (uint64_t *)fs->volume;
  uint64_t pp = p[n];
  int num = 0;
  for (int i = 63; i >= 0; i--) {
    if (((pp >> i) & 1L) == 0) {
      num++;
    } else {
      break;
    }
  }
  return num;
}

__device__ int start_consecutive_nblk(FileSystem *fs, int n) {
  uint64_t *p = (uint64_t *)fs->volume;
  uint64_t pp = p[n];
  int num = 0;
  for (int i = 0; i <= 63; i++) {
    if (((pp >> i) & 1L) == 0) {
      num++;
    } else {
      break;
    }
  }
  return num;
}

// Get n consecutive blocks
// a block is 32bytes
__device__ int get_empty_block(FileSystem *fs, int n) {
  uint64_t *p = (uint64_t *)fs->volume;
  for (int i = 0; i < fs->SUPERBLOCK_SIZE / sizeof(uint64_t); i++) {
    if (p[i] == UINT64_MAX)
      continue;
    int probe_start = ffsl(~p[i]);
    if (probe_start + n - 1 >= 64 &&
        i < fs->SUPERBLOCK_SIZE / sizeof(uint64_t) - 1) {
      if (end_consecutive_nblk(fs, i) + start_consecutive_nblk(fs, i + 1) >=
          n) {
        return get_block_from_2_unit(fs, i, i + 1, n);
      }
    } else {
      for (int k = probe_start; k <= 64 - n; k++) {
        uint64_t negate = ~p[i];
        if (((negate >> k) & MASK[n]) == MASK[n]) {
          // find it
          p[i] |= (MASK[n] << k);
          return i * 64 + k;
        }
      }
      if (i < fs->SUPERBLOCK_SIZE / sizeof(uint64_t) - 1 &&
          end_consecutive_nblk(fs, i) + start_consecutive_nblk(fs, i + 1) >=
              n) {
        return get_block_from_2_unit(fs, i, i + 1, n);
      }
    }
  }
  // no empty block, maybe need compact
  return -1;
}

// order by modified time of files
__device__ int cmpLS_D(FileSystem *fs, int i, int j) {
  FCB *fcb1 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
  FCB *fcb2 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j);
  if (fcb1->m_time < fcb2->m_time)
    return 1;
  else if (fcb1->m_time == fcb2->m_time)
    return 0;
  else
    return -1;
}

// order by size
__device__ int cmpLS_S(FileSystem *fs, int i, int j) {
  FCB *fcb1 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
  FCB *fcb2 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j);
  if (fcb1->file_sz < fcb2->file_sz)
    return 1;
  else if (fcb1->file_sz == fcb2->file_sz)
    return 0;
  else
    return -1;
}

__device__ void bubble_sort(FileSystem *fs, int *array, int len, CmpFunc cmp) {
  int tem;
  for (int i = 0; i < len - 1; i++) {
    for (int j = 0; j < len - 1 - i; j++) {
      if (cmp(fs, array[j], array[j + 1]) > 0) {
        tem = array[j];
        array[j] = array[j + 1];
        array[j + 1] = tem;
      }
    }
  }
}

// Calculate the number of blocks occupied by the file_sz file size
__device__ int compute_nblk(FileSystem *fs, int file_sz) {
  int nblk;
  if (file_sz == 0) {
    nblk = 0;
  } else {
    nblk = (file_sz - 1) / fs->STORAGE_BLOCK_SIZE + 1;
  }
  return nblk;
}

__device__ void clear_super_blk(FileSystem *fs, int addr, int nblk) {
  if (nblk == 0)
    return;
  uint64_t *p = (uint64_t *)fs->volume;
  int idx = addr / 64;
  int off = addr % 64;
  p[idx] &= ~(MASK[nblk] << off);
  addr2fd[addr] = -1;
}

__device__ int ffsl(uint64_t num) {
  for (int i = 0; i < 64; i++) {
    if (num & (1L << i)) {
      return i;
    }
  }
  return -1;
}

__device__ int get_block_from_2_unit(FileSystem *fs, int n1, int n2, int nblk) {
  int nblk1 = end_consecutive_nblk(fs, n1);
  int nblk2 = nblk - nblk1;
  uint64_t *p = (uint64_t *)fs->volume;
  p[n1] |= (MASK[nblk1] << (64 - nblk1));
  p[n2] |= MASK[nblk2];
  return 64 * n1 + 64 - nblk1;
}

__device__ int consecutive_nblk(FileSystem *fs, int n, int idx) {
  uint64_t *p = (uint64_t *)fs->volume;
  uint64_t pp = p[n];
  int num = 0;
  for (int i = idx; i <= 63; i++) {
    if (((pp >> i) & 1L) == 0) {
      num++;
    } else {
      break;
    }
  }
  return num;
}

__device__ int the_longest_consecutive_nonempty_nblk(FileSystem *fs, int n,
                                                     int idx) {
  uint64_t *p = (uint64_t *)fs->volume;
  uint64_t pp = p[n];
  int num = 0;
  for (int i = idx; i <= 63; i++) {
    if (((pp >> i) & 1L) == 1) {
      num++;
    } else {
      break;
    }
  }
  for (int i = n + 1; i < fs->SUPERBLOCK_SIZE / sizeof(uint64_t); i++) {
    if (p[i] == UINT64_MAX) {
      num += 64;
      continue;
    }
    uint64_t pp = p[i];
    for (int k = 0; k <= 63; k++) {
      if (((pp >> k) & 1L) == 1) {
        num++;
      } else {
        return num;
      }
    }
  }
  return num;
}

__device__ void move_storage(FileSystem *fs, int bid1, int bid2) {
  int storage_start_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->FCB_ENTRIES;
  uchar *b1 = &fs->volume[storage_start_addr + bid1 * fs->STORAGE_BLOCK_SIZE];
  uchar *b2 = &fs->volume[storage_start_addr + bid2 * fs->STORAGE_BLOCK_SIZE];
  for (int i = 0; i < 32; i++) {
    b1[i] = b2[i];
  }
}

__device__ void set_bitmap(uchar* p, int k) {
  *p |= (1<<k);
}

__device__ void clear_bitmap(uchar* p, int k) {
  *p &= ~(1<<k);
}

__device__ void compact(FileSystem *fs) {
  // printf("compact\n");
  uint64_t *p = (uint64_t *)fs->volume;
  for (int i = 0; i < fs->SUPERBLOCK_SIZE / sizeof(uint64_t); i++) {
    if (p[i] == UINT64_MAX)
      continue;
    int old_pos = -1;
    while (1) {
      int pos = ffsl(~p[i]);
      if (pos == -1 || old_pos == pos) {
        // printf("break i %d\n", i);
        break;
      }
      old_pos = pos;
      int to_move_nblk = consecutive_nblk(fs, i, pos);
      int longest_nonempty_nblk =
          the_longest_consecutive_nonempty_nblk(fs, i, pos + to_move_nblk);
      // move
      int to_move_start = i * 64 + pos + to_move_nblk;
      // printf("to_move_nblk %d longest_nonempty_nblk %d to_move_start: %d pos "
      //        "%d old_pos %d\n",
      //        to_move_nblk, longest_nonempty_nblk, to_move_start, pos, old_pos);
      for (int k = to_move_start; k < to_move_start + longest_nonempty_nblk;
           k++) {
        set_bitmap((uchar*)&fs->volume[(k - to_move_nblk)/8], (k - to_move_nblk)%8);
        move_storage(fs, k - to_move_nblk, k);
      }
      for (int k = to_move_start + longest_nonempty_nblk - to_move_nblk;
           k < to_move_start + longest_nonempty_nblk; k++) {
        clear_bitmap((uchar*)&fs->volume[k/8], k%8);
      }
      // printf("move done\n");
      // update fcb's addr
      for (int k = to_move_start; k < to_move_start + longest_nonempty_nblk;
           k++) {
        if (addr2fd[k] != -1) {
          int fd = addr2fd[k];
          FCB *fcb =
              AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fd);
          fcb->addr = k - to_move_nblk;
          addr2fd[k] = -1;
          addr2fd[k - to_move_nblk] = fd;
        }
      }
      // printf("update map done\n");
    }
  }
}

// ***************************** Static Allocation Implementation*************************************

// #include "file_system.h"
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <stdlib.h>

// // helper function declaration
// __device__ size_t my_strlen(const char *str);
// __device__ int my_strcmp(const char *str1, const char *str2);
// __device__ char *my_strcpy(char *dst, const char *src);
// __device__ int get_empty_block(FileSystem *fs);
// __device__ int cmpLS_D(FileSystem *fs, int i, int j);
// __device__ int cmpLS_S(FileSystem *fs, int i, int j);
// typedef int (*CmpFunc)(FileSystem *, int, int);
// __device__ void bubble_sort(FileSystem *fs, int *array, int len, CmpFunc cmp);
// __device__ __managed__ u32 gtime = 0;

// /**
//  * superblock: Volume control block (master file table) contains volume details
//  * File Control Block (FCB): contains many details about the file
//  */
// // global structure of volume
// // super block(4KB) / FCB(32KB) / contents of the files(1024KB)
// // The maximum size of a file name is 20 bytes
// #define MAX_FILE_NAME_LEN 20
// #define INVALID UINT32_MAX

// // the structure of File Control Block
// typedef struct FCB {
//   char file_name[MAX_FILE_NAME_LEN]; // The maximum size of a file name is 20
//                                      // bytes.
//   uint16_t m_time;                   // modify time
//   uint16_t c_time;                   // create time
//   uint32_t file_sz;                  // file size
//   uint32_t addr;                     // location of the file contents
// } FCB;

// #define AddrToFCB(addr) ((FCB *)(addr))

// __device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
//                         int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
//                         int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
//                         int MAX_FILE_NUM, int MAX_FILE_SIZE,
//                         int FILE_BASE_ADDRESS) {
//   // init variables
//   fs->volume = volume;

//   // init constants
//   fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
//   fs->FCB_SIZE = FCB_SIZE;
//   fs->FCB_ENTRIES = FCB_ENTRIES;
//   fs->STORAGE_SIZE = VOLUME_SIZE;
//   fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
//   fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
//   fs->MAX_FILE_NUM = MAX_FILE_NUM;
//   fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
//   fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

//   memset(volume, 0, VOLUME_SIZE);
// }

// // Because the maximum number of files is 1024, the maximum size of a file is
// // 1024 bytes, and the storage can just hold all of them, so can directly
// // allocate 32 consecutive blocks to each file at one time.
// __device__ u32 fs_open(FileSystem *fs, char *s, int op) {
//   /* Implement open operation here */
//   int first_empty_file_pos = -1;
//   for (int i = 0; i < fs->FCB_ENTRIES; i++) {
//     // Get the address of the i-th FCB
//     FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
//     // compare filename
//     if (first_empty_file_pos == -1 && my_strlen(fcb->file_name) == 0) {
//       first_empty_file_pos = i;
//     }
//     if (my_strcmp(s, fcb->file_name) == 0) {
//       return i;
//     }
//   }
//   // this file not exists
//   if (op == G_WRITE) {
//     if (first_empty_file_pos == -1) {
//       // no empty file
//       return INVALID;
//     }
//     // create a new zero byte file.
//     // allocate a new empty block
//     FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE +
//                          fs->FCB_SIZE * first_empty_file_pos);
//     int pos = get_empty_block(fs);

//     if (pos != -1) {
//       fcb->addr = pos;
//     } else {
//       return INVALID;
//     }
//     my_strcpy(fcb->file_name, s);
//     fcb->file_sz = 0;
//     fcb->c_time = gtime;
//     fcb->m_time = gtime;
//     gtime++;
//     return first_empty_file_pos;
//   } else if (op == G_READ) {
//     return INVALID;
//   } else {
//     // should not run here
//     return INVALID;
//   }
// }

// __device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp) {
//   /* Implement read operation here */
//   if (fp == INVALID || fp >= fs->MAX_FILE_NUM) {
//     return;
//   }

//   FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp);
//   if (my_strlen(fcb->file_name) == 0 || size > fcb->file_sz) {
//     return;
//   }

//   int storage_start_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->FCB_ENTRIES;
//   uchar *to_read =
//       &fs->volume[storage_start_addr + fcb->addr * fs->STORAGE_BLOCK_SIZE];
//   for (int i = 0; i < size; i++) {
//     output[i] = to_read[i];
//   }
// }

// __device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp) {
//   /* Implement write operation here */
//   if (fp == INVALID || fp >= fs->MAX_FILE_NUM) {
//     return INVALID;
//   }

//   FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp);
//   if (my_strlen(fcb->file_name) == 0) {
//     return INVALID;
//   }

//   fcb->file_sz = size;
//   fcb->m_time = gtime++;
//   int storage_start_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->FCB_ENTRIES;
//   uchar *to_write =
//       &fs->volume[storage_start_addr + fcb->addr * fs->STORAGE_BLOCK_SIZE];
//   for (int i = 0; i < size; i++) {
//     to_write[i] = input[i];
//   }
//   return 0;
// }

// __device__ void fs_gsys(FileSystem *fs, int op) {
//   /* Implement LS_D and LS_S operation here */
//   int fcbs[1024];
//   int num = 0;
//   for (int i = 0; i < fs->FCB_ENTRIES; i++) {
//     FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
//     if (my_strlen(fcb->file_name) != 0) {
//       fcbs[num++] = i;
//     }
//   }

//   if (op == LS_D) {
//     printf("===sort by modified time===\n");
//     bubble_sort(fs, fcbs, num, cmpLS_D);
//     for (int i = 0; i < num; i++) {
//       FCB *fcb =
//           AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fcbs[i]);
//       printf("%s\n", fcb->file_name);
//     }
//   } else if (op == LS_S) {
//     printf("===sort by file size===\n");
//     bubble_sort(fs, fcbs, num, cmpLS_S);
//     for (int i = 0; i < num; i++) {
//       FCB *fcb =
//           AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fcbs[i]);
//       printf("%s %d\n", fcb->file_name, fcb->file_sz);
//     }
//   } else {
//     // should not run here
//     return;
//   }
// }

// __device__ void fs_gsys(FileSystem *fs, int op, char *s) {
//   /* Implement rm operation here */
//   if (op != RM)
//     return;
//   for (int i = 0; i < fs->FCB_ENTRIES; i++) {
//     // Get the address of the i-th FCB
//     FCB *fcb = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
//     // compare filename
//     if (my_strcmp(s, fcb->file_name) == 0) {
//       // clear super block
//       int *p = (int *)fs->volume;
//       p[fcb->addr / 32] = 0;
//       // clear fcb, I judge by whether the filename is empty
//       fcb->file_name[0] = 0;
//     }
//   }
// }

// /****************************** helper function ******************************/
// __device__ size_t my_strlen(const char *str) {
//   const char *eos = str;
//   while (*eos++)
//     ;
//   return (eos - str - 1);
// }

// __device__ int my_strcmp(const char *str1, const char *str2) {
//   int ret = 0;
//   while (!(ret = *(unsigned char *)str1 - *(unsigned char *)str2) && *str1) {
//     str1++;
//     str2++;
//   }

//   if (ret < 0) {
//     return -1;
//   } else if (ret > 0) {
//     return 1;
//   }
//   return 0;
// }

// __device__ char *my_strcpy(char *dst, const char *src) {
//   char *ret = dst;
//   while ((*dst++ = *src++) != '\0')
//     ;
//   return ret;
// }

// // 32 blocks are allocated at one time, so the unit is 4 bytes. If the int value
// // corresponding to these 4 bytes is 1, it means that these 32 consecutive
// // blocks have been allocated.
// __device__ int get_empty_block(FileSystem *fs) {
//   int *p = (int *)fs->volume;
//   for (int i = 0; i < fs->SUPERBLOCK_SIZE / 4; i++) {
//     if (p[i] != 1) {
//       p[i] = 1;
//       return i * 32;
//     }
//   }
//   // no empty block
//   return -1;
// }

// // order by modified time of files
// __device__ int cmpLS_D(FileSystem *fs, int i, int j) {
//   FCB *fcb1 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
//   FCB *fcb2 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j);
//   if (fcb1->m_time < fcb2->m_time)
//     return 1;
//   else if (fcb1->m_time == fcb2->m_time)
//     return 0;
//   else
//     return -1;
// }

// // order by size
// __device__ int cmpLS_S(FileSystem *fs, int i, int j) {
//   FCB *fcb1 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i);
//   FCB *fcb2 = AddrToFCB(fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j);
//   if (fcb1->file_sz < fcb2->file_sz)
//     return 1;
//   else if (fcb1->file_sz == fcb2->file_sz)
//     return 0;
//   else
//     return -1;
// }

// __device__ void bubble_sort(FileSystem *fs, int *array, int len, CmpFunc cmp) {
//   int tem;
//   for (int i = 0; i < len - 1; i++) {
//     for (int j = 0; j < len - 1 - i; j++) {
//       if (cmp(fs, array[j], array[j + 1]) > 0) {
//         tem = array[j];
//         array[j] = array[j + 1];
//         array[j + 1] = tem;
//       }
//     }
//   }
// }