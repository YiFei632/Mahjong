# file: model_pool.py
from multiprocessing.shared_memory import SharedMemory, ShareableList
import _pickle as cPickle
import time
from collections import deque # 导入 deque

class ModelPoolServer:
    
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 1024
        self.shared_model_list = ShareableList([' ' * metadata_size] * capacity + [self.n], name = name)
        
        # 新增: 创建一个队列来管理待清理的旧共享内存对象
        # 其大小设为 capacity，确保我们至少保留一代旧模型，为 Actor 留出充足的加载时间
        self.cleanup_queue = deque()

    def push(self, state_dict, metadata = {}):
        n = self.n % self.capacity

        # --- 修改开始 ---
        # 1. 如果当前位置n有一个旧模型，我们不再立即unlink它
        # 而是将其对应的SharedMemory对象放入待清理队列
        if self.model_list[n]:
            old_memory_obj = self.model_list[n]['memory']
            self.cleanup_queue.append(old_memory_obj)

        # 2. 当待清理队列过长时（例如，超过缓冲区大小），
        # 我们就认为队列最头部的那个内存对象已经足够老，可以安全地删除了
        if len(self.cleanup_queue) > self.capacity:
            memory_to_delete = self.cleanup_queue.popleft()
            try:
                # 在 unlink 前先 close
                memory_to_delete.close()
                memory_to_delete.unlink()
                # print(f"Cleaned up stale model's shared memory: {memory_to_delete.name}")
            except FileNotFoundError:
                # 如果文件已经被其他方式清理，忽略错误
                # print(f"Stale model's shared memory {memory_to_delete.name} was already gone.")
                pass
        # --- 修改结束 ---

        data = cPickle.dumps(state_dict) # model parameters serialized to bytes
        memory = SharedMemory(create = True, size = len(data))
        memory.buf[:] = data[:]
        
        metadata = metadata.copy()
        metadata['_addr'] = memory.name
        metadata['id'] = self.n
        
        # 将 SharedMemory 对象本身也存起来，以便后续清理
        metadata['memory'] = memory
        self.model_list[n] = metadata

        # 将不包含 memory 对象的元数据序列化
        serializable_metadata = {k: v for k, v in metadata.items() if k != 'memory'}
        self.shared_model_list[n] = cPickle.dumps(serializable_metadata)

        self.n += 1
        self.shared_model_list[-1] = self.n

    # 新增: 增加一个清理方法，在程序结束时调用，确保所有内存都被释放
    def shutdown(self):
        print("ModelPoolServer shutting down. Cleaning up all shared memory...")
        # 清理仍在 model_list 中的内存
        for model_meta in self.model_list:
            if model_meta and 'memory' in model_meta:
                try:
                    model_meta['memory'].close()
                    model_meta['memory'].unlink()
                except FileNotFoundError:
                    pass
        
        # 清理所有待清理队列中的内存
        while self.cleanup_queue:
            memory_to_delete = self.cleanup_queue.popleft()
            try:
                memory_to_delete.close()
                memory_to_delete.unlink()
            except FileNotFoundError:
                pass
        
        # 清理 ShareableList 本身
        self.shared_model_list.shm.close()
        self.shared_model_list.shm.unlink()
        print("ModelPoolServer cleanup complete.")


class ModelPoolClient:
    
    def __init__(self, name):
        # 增加重试逻辑以应对 Learner 尚未创建 ShareableList 的情况
        while True:
            try:
                self.shared_model_list = ShareableList(name = name)
                # 确认可以访问
                _ = self.shared_model_list[-1]
                break
            except (FileNotFoundError, IndexError):
                # print(f"Waiting for ModelPoolServer ({name}) to start...")
                time.sleep(1)

        self.capacity = len(self.shared_model_list) - 1
        self.model_list = [None] * self.capacity
        self.n = 0
        self._update_model_list()
    
    def _update_model_list(self):
        try:
            n = self.shared_model_list[-1]
            if n > self.n:
                # new models available, update local list
                for i in range(max(self.n, n - self.capacity), n):
                    self.model_list[i % self.capacity] = cPickle.loads(self.shared_model_list[i % self.capacity])
                self.n = n
        except FileNotFoundError:
            print("ModelPool has been shut down. Can't update model list.")
            # 当服务器关闭后，n 可能无法访问
            pass

    def get_model_list(self):
        self._update_model_list()
        model_list = []
        start_index = self.n % self.capacity if self.n >= self.capacity else 0
        
        # 从起始索引到列表末尾
        model_list.extend([m for m in self.model_list[start_index:] if m is not None])
        # 从列表开头到起始索引
        model_list.extend([m for m in self.model_list[:start_index] if m is not None])
        
        return model_list
    
    def get_latest_model(self):
        self._update_model_list()
        while self.n == 0:
            time.sleep(0.5) # 稍微增加等待时间
            self._update_model_list()
        # 从本地缓存中获取最新的有效模型元数据
        latest_model_meta = self.model_list[(self.n - 1) % self.capacity]
        return latest_model_meta
        
    def load_model(self, metadata):
        if not metadata or '_addr' not in metadata:
            return None # 如果元数据无效，则返回
        
        # 在加载前最后更新一次模型列表，以防万一
        self._update_model_list()
        n = metadata['id']
        # 如果模型ID太旧，说明它已经被清理了
        if n < self.n - self.capacity * 2: # 使用更保守的检查
            # print(f"Model {n} is too old, likely cleaned up. Current is {self.n}.")
            return None
        
        try:
            memory = SharedMemory(name = metadata['_addr'])
            state_dict = cPickle.loads(memory.buf)
            memory.close() # 读取后立即关闭连接
            return state_dict
        except FileNotFoundError:
            # print(f"Could not find shared memory for model {n} ({metadata['_addr']}). It might have been cleaned up.")
            return None # 如果加载时文件不存在，优雅地返回None

    # 新增: Client也需要一个关闭方法
    def shutdown(self):
        self.shared_model_list.shm.close()