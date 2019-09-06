class Nodes:
    alias_1702 = '1702'
    alias_gpu = 'gpu'
    alias_cpu = 'cpu'

    uname_1702 = 'aida-lab-1702'
    uname_gpu = 'AIDA1080Ti'
    uname_cpu = 'compute-AIDA14-ubuntu'

    uname_cache = alias_cache = None
    supported_pairs = [(uname_1702, alias_1702), (uname_gpu, alias_gpu), (uname_cpu, alias_cpu)]
    dft = object()

    @staticmethod
    def is_alias_supported(node_alias):
        return node_alias in {Nodes.alias_1702, Nodes.alias_gpu, Nodes.alias_cpu}

    @staticmethod
    def get_uname():
        if Nodes.uname_cache is None:
            from subprocess import Popen, PIPE as P
            sub = Popen("uname -a", stdin=P, stdout=P, stderr=P, shell=True, bufsize=1)
            out_bytes, _ = sub.communicate()
            Nodes.uname_cache = str(out_bytes, encoding='utf8')
        return Nodes.uname_cache

    @staticmethod
    def get_alias():
        if Nodes.alias_cache is None:
            uname_full = Nodes.get_uname()
            for uname_part, alias in Nodes.supported_pairs:
                if uname_part in uname_full:
                    Nodes.alias_cache = alias
                    print('USE SERVER @ {}'.format(Nodes.alias_cache))
                    return alias
            if Nodes.alias_cache is None:
                raise ValueError('unsupported node: "{}"'.format(uname_full))
        return Nodes.alias_cache

    @staticmethod
    def select(n1702=dft, ngpu=dft, ncpu=dft, default=dft):
        table = {Nodes.alias_1702: n1702, Nodes.alias_gpu: ngpu, Nodes.alias_cpu: ncpu}
        result = table.get(Nodes.get_alias())
        if result is not Nodes.dft:
            return result
        else:
            if default is Nodes.dft:
                raise ValueError('[{}] uses default value, but no default value is given'.
                                 format(Nodes.get_alias()))
            else:
                return default

    @staticmethod
    def max_cpu_num():
        return Nodes.select(n1702=12, ngpu=32, ncpu=20)

    @staticmethod
    def is_1702():
        return Nodes.get_alias() == Nodes.alias_1702

    @staticmethod
    def is_gpu():
        return Nodes.get_alias() == Nodes.alias_gpu

    @staticmethod
    def is_cpu():
        return Nodes.get_alias() == Nodes.alias_cpu
