class PrintingObject:

    class SilenceSignal():
        def __init__(self, obj, value):
            self.obj = obj
            self.new_silent = value
        def __enter__(self):
            self.old_silent = self.obj.get_silence()
            self.obj.set_silence(self.new_silent)
        def __exit__(self, exception_type, exception_value, traceback):
            self.obj.set_silence(self.old_silent)
    
    def __init__(self):
        self.silent = True
    
    def is_silent(self):
        return self.silent
    
    def get_silence(self):
        return self.is_silent()
    
    def set_silence(self, value=True):
        self.silent = value
        return self
    
    def unset_silence(self):
        self.silent = False
        return self
        
    def with_silence(self, value=True):
        self.set_silence(value)
        return self
        
    def silence(self, value=True):
        return self.__class__.SilenceSignal(self, value)
        
    def _print(self, *args, **kwargs):
        if not self.silent:
            print(*args, **kwargs)