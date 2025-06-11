from .special import SScalar



class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
    def register_parameter(self, name: str, param: SScalar):
        self._parameters[name] = param
    
    def register_module(self, name: str, module: 'Module'):
        self._modules[name] = module
    
    def __setattr__(self, name, value):
        if isinstance(value, SScalar):
            if hasattr(self, '_parameters'):
                self._parameters[name] = value
        elif isinstance(value, Module):
            if hasattr(self, '_modules'):
                self._modules[name] = value
        # always call parent to actually set the attribute
        super().__setattr__(name, value)
    
    def parameters(self):
        params = []
        
        # add parameters from this module
        if hasattr(self, '_parameters'):
            params.extend(self._parameters.values())
        
        # add parameters from submodules
        if hasattr(self, '_modules'):
            for module in self._modules.values():
                params.extend(module.parameters())
        
        # special cases -- like lists of parameters or modules
        for attr_name in self.__dict__:
            attr = getattr(self, attr_name)
            if isinstance(attr, list) and attr:
                if isinstance(attr[0], SScalar):
                    params.extend(attr)
                elif isinstance(attr[0], Module):
                    for module in attr:
                        params.extend(module.parameters())
        
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self.modules = modules or []
        # register each module
        for i, module in enumerate(self.modules):
            self.register_module(str(i), module)
    
    def append(self, module):
        self.modules.append(module)
        self.register_module(str(len(self.modules) - 1), module)
    
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __len__(self):
        return len(self.modules)
    
    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
