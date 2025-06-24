from multiprocessing.shared_memory import SharedMemory, ShareableList
import _pickle as cPickle
import time
import atexit

class ModelPoolServer:
    
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        self.memory_objects = []  # 追踪所有创建的内存对象
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 1024
        self.shared_model_list = ShareableList([' ' * metadata_size] * capacity + [self.n], name = name)
        
        # 注册清理函数
        atexit.register(self.cleanup)
        
    def push(self, state_dict, metadata = {}):
        n = self.n % self.capacity
        
        # 不要立即删除旧内存，让它们自然过期
        # 只有在容量满了且循环一圈后才删除最老的
        if len(self.memory_objects) >= self.capacity * 2:  # 保留两轮的内存
            old_memory = self.memory_objects.pop(0)
            try:
                old_memory.close()
                old_memory.unlink()
            except:
                pass
        
        try:
            data = cPickle.dumps(state_dict)
            memory = SharedMemory(create = True, size = len(data))
            memory.buf[:] = data[:]
            
            # 追踪内存对象
            self.memory_objects.append(memory)
            
            metadata = metadata.copy()
            metadata['_addr'] = memory.name
            metadata['id'] = self.n
            
            self.model_list[n] = metadata
            self.shared_model_list[n] = cPickle.dumps(metadata)
            
            self.n += 1
            self.shared_model_list[-1] = self.n
            
        except Exception as e:
            print(f"Error in ModelPoolServer.push: {e}")
    
    def cleanup(self):
        """程序退出时的清理函数"""
        try:
            # 清理所有内存对象
            for memory in self.memory_objects:
                try:
                    memory.close()
                    memory.unlink()
                except:
                    pass
            
            # 清理ShareableList
            try:
                self.shared_model_list.shm.close()
                self.shared_model_list.shm.unlink()
            except:
                pass
        except:
            pass

class ModelPoolClient:
    
    def __init__(self, name):
        self.name = name
        self.shared_model_list = None
        
        # 连接到共享列表，增加重试逻辑
        max_retries = 30
        for attempt in range(max_retries):
            try:
                self.shared_model_list = ShareableList(name = name)
                n = self.shared_model_list[-1]
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to ModelPool after {max_retries} attempts")
                time.sleep(0.5)
        
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
                    try:
                        data = self.shared_model_list[i % self.capacity]
                        if data and data.strip():  # 检查数据是否有效
                            self.model_list[i % self.capacity] = cPickle.loads(data)
                    except:
                        continue
                self.n = n
        except:
            pass
    
    def get_model_list(self):
        self._update_model_list()
        model_list = []
        if self.n >= self.capacity:
            model_list.extend([m for m in self.model_list[self.n % self.capacity :] if m is not None])
        model_list.extend([m for m in self.model_list[: self.n % self.capacity] if m is not None])
        return model_list
    
    def get_latest_model(self):
        self._update_model_list()
        
        # 增加超时机制
        max_wait = 30  # 最多等待30秒
        wait_time = 0
        while self.n == 0 and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1
            self._update_model_list()
        
        if self.n == 0:
            return None
            
        return self.model_list[(self.n + self.capacity - 1) % self.capacity]
        
    def load_model(self, metadata):
        if not metadata or '_addr' not in metadata:
            return None
            
        try:
            self._update_model_list()
            n = metadata['id']
            
            # 检查模型是否太旧
            if n < self.n - self.capacity * 2:
                return None
                
            memory = SharedMemory(name = metadata['_addr'])
            state_dict = cPickle.loads(memory.buf)
            memory.close()  # 立即关闭
            return state_dict
            
        except Exception as e:
            # print(f"Failed to load model {metadata.get('id', 'unknown')}: {e}")
            return None
    
    def close(self):
        """客户端关闭时调用"""
        try:
            if self.shared_model_list:
                self.shared_model_list.shm.close()
        except:
            pass