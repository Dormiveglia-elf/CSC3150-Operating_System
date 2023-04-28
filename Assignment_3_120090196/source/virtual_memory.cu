#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define PAGE_SZ 32
#define PAGE_BIT 5
#define INVALID 0x80000000
#define VALID_MASK 0x7FFFFFFF

__device__ int *lru_replacer;
__device__ long int time_stamp = 0;
struct PageAddr {
  u32 page_id;
  u32 page_off;
};

// parse addr
__device__ struct PageAddr parse_addr(u32 addr) {
  struct PageAddr page_addr;
  page_addr.page_id = addr / PAGE_SZ;
  page_addr.page_off = addr % PAGE_SZ;
  return page_addr;
}

// if find, return frame_id, else return -1
__device__ int invert_page_table_find(VirtualMemory *vm, u32 page_id) {
  int idx;
  for (idx = 1; idx <= vm->PAGE_ENTRIES; idx++) {
    if (vm->invert_page_table[idx - 1 + vm->PAGE_ENTRIES] == page_id) {
      if ((vm->invert_page_table[idx - 1] & INVALID) == 0) {
        // find it
        return idx;
      }
      // invalid
      return -idx;
    }
  }
  // not in invert_page_table
  return 0;
}

// copy PAGE_SIZE data from storage to physical memory
__device__ void copy_from_storage_2_vm(VirtualMemory *vm, int frame_id,
                                       int page_id) {
  u32 phy_base = frame_id << 5;
  u32 storage_base = page_id << 5;
  for (int i = 0; i < PAGE_SZ; i++) {
    vm->buffer[phy_base + i] = vm->storage[storage_base + i];
  }
}

// flush PAGE_SIZE data from physical memory to storage
__device__ void flush_data_from_vm_2_storage(VirtualMemory *vm, int frame_id,
                                             int page_id) {
  u32 phy_base = frame_id << 5;
  u32 storage_base = page_id << 5;
  for (int i = 0; i < PAGE_SZ; i++) {
    vm->storage[storage_base + i] = vm->buffer[phy_base + i];
  }
}

__device__ void update_lru(int frame_id) {
  time_stamp++;
  lru_replacer[frame_id] = time_stamp;
}

__device__ int victim(VirtualMemory *vm) {
  int min = INT32_MAX;
  int min_idx = -1;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (lru_replacer[i] < min) {
      min = lru_replacer[i];
      min_idx = i;
    }
  }
  return min_idx;
}

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i] |= threadIdx.x;
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);

  // init lru
  lru_replacer = (int *)malloc(vm->PAGE_ENTRIES * sizeof(int));
  memset(lru_replacer, 0, vm->PAGE_ENTRIES * sizeof(int));
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  // parse addr
  struct PageAddr page_addr = parse_addr(addr);
  int frame_id = invert_page_table_find(vm, page_addr.page_id);
  if (frame_id < 0) {
    // just invalid
    frame_id = -frame_id - 1;
    // page fault
    *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;
    // update page_table
    vm->invert_page_table[frame_id] &= VALID_MASK; // valid
    vm->invert_page_table[frame_id + vm->PAGE_ENTRIES] =
        page_addr.page_id; // frame_id <---> page_id
    // read it from storage
    copy_from_storage_2_vm(vm, frame_id, page_addr.page_id);
    // update lru
    update_lru(frame_id);
    // get physical addr
    u32 phy_addr = (frame_id << 5) + page_addr.page_off;
    return vm->buffer[phy_addr];
  } else if (frame_id == 0) {
    // replace page
    // page fault
    *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;
    // find Least Recently Used frame
    int victim_frame_id = victim(vm);
    // flush data from memory to storage
    flush_data_from_vm_2_storage(
        vm, victim_frame_id,
        vm->invert_page_table[victim_frame_id + vm->PAGE_ENTRIES]);
    // read it from storage
    copy_from_storage_2_vm(vm, victim_frame_id, page_addr.page_id);
    vm->invert_page_table[victim_frame_id] &= VALID_MASK; // set valid
    vm->invert_page_table[victim_frame_id + vm->PAGE_ENTRIES] =
        page_addr.page_id; // frame_id <---> page_id
    // update lru
    update_lru(victim_frame_id);
    // get physical addr
    u32 phy_addr = (victim_frame_id << 5) + page_addr.page_off;
    return vm->buffer[phy_addr];
  } else {
    frame_id--;
    // hit
    // update lru
    update_lru(frame_id);
    // get physical addr
    u32 phy_addr = (frame_id << 5) + page_addr.page_off;
    return vm->buffer[phy_addr];
  }
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  // parse addr
  struct PageAddr page_addr = parse_addr(addr);
  int frame_id = invert_page_table_find(vm, page_addr.page_id);
  if (frame_id < 0) {
    // just invalid
    frame_id = -frame_id - 1;
    // page fault
    *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;
    // update page_table
    vm->invert_page_table[frame_id] &= VALID_MASK; // valid
    vm->invert_page_table[frame_id + vm->PAGE_ENTRIES] =
        page_addr.page_id; // frame_id <---> page_id
    // read it from storage
    copy_from_storage_2_vm(vm, frame_id, page_addr.page_id);
    // update lru
    update_lru(frame_id);
    // get physical addr
    u32 phy_addr = (frame_id << 5) + page_addr.page_off;
    vm->buffer[phy_addr] = value;
    return;
  } else if (frame_id == 0) {
    // replace page
    // page fault
    *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;
    // find Least Recently Used frame
    int victim_frame_id = victim(vm);
    // flush data from memory to storage
    flush_data_from_vm_2_storage(
        vm, victim_frame_id,
        vm->invert_page_table[victim_frame_id + vm->PAGE_ENTRIES]);
    // read it from storage
    copy_from_storage_2_vm(vm, victim_frame_id, page_addr.page_id);
    vm->invert_page_table[victim_frame_id] &= VALID_MASK; // set valid
    vm->invert_page_table[victim_frame_id + vm->PAGE_ENTRIES] =
        page_addr.page_id; // frame_id <---> page_id
    // update lru
    update_lru(victim_frame_id);
    // get physical addr
    u32 phy_addr = (victim_frame_id << 5) + page_addr.page_off;
    vm->buffer[phy_addr] = value;
    return;
  } else {
    frame_id--;
    // hit
    // update lru
    update_lru(frame_id);
    // get physical addr
    u32 phy_addr = (frame_id << 5) + page_addr.page_off;
    vm->buffer[phy_addr] = value;
    return;
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int idx = 0; idx < input_size; idx++) {
    results[idx] = vm_read(vm, idx + offset);
  }
}
