Search.setIndex({"docnames": ["api/empty_workspace_list", "api/index", "api/methods", "api/operators", "api/traits", "examples", "examples/examples_matrix", "index", "installation", "introduction1", "introduction2", "license", "performance", "release_notes", "requirements_func", "warnings/bit_identical", "warnings/bit_identical_sm", "warnings/msvc"], "filenames": ["api/empty_workspace_list.rst", "api/index.rst", "api/methods.rst", "api/operators.rst", "api/traits.rst", "examples.rst", "examples/examples_matrix.rst", "index.rst", "installation.rst", "introduction1.rst", "introduction2.rst", "license.rst", "performance.rst", "release_notes.rst", "requirements_func.rst", "warnings/bit_identical.rst", "warnings/bit_identical_sm.rst", "warnings/msvc.rst"], "titles": ["&lt;no title&gt;", "cuFFTDx API Reference", "Execution Methods", "Operators", "Traits", "Examples", "&lt;no title&gt;", "NVIDIA cuFFTDx", "Quick Installation Guide", "First FFT Using cuFFTDx", "Your Next Custom FFT Kernels", "Software License Agreement", "Achieving High Performance", "Release Notes", "Requirements and Functionality", "&lt;no title&gt;", "&lt;no title&gt;", "&lt;no title&gt;"], "terms": {"workspac": [0, 1, 5, 9, 10, 14], "i": [0, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16], "requir": [0, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 17], "fft": [0, 2, 6, 7, 12, 13, 14, 15, 16], "follow": [0, 2, 3, 4, 8, 11, 14], "size": [0, 2, 5, 7, 9, 10, 12, 13, 14, 15], "power": [0, 2, 3, 13, 14], "2": [0, 2, 3, 4, 5, 9, 11, 12, 14], "up": [0, 2, 5, 10, 12, 14], "32768": [0, 2, 3, 4, 14], "3": [0, 2, 5, 7, 9, 11, 12, 14], "19683": [0, 2, 14], "5": [0, 2, 11, 12, 14], "15625": [0, 2, 14], "6": [0, 2, 4, 11, 12, 14], "1296": [0, 2, 14], "7": [0, 2, 11, 12, 14], "2401": [0, 2, 14], "10": [0, 2, 8, 11, 14], "10000": [0, 2, 14], "11": [0, 2, 14], "1331": [0, 2, 14], "12": [0, 2, 14], "1728": [0, 2, 14], "In": [0, 5, 7, 9, 10, 12, 13, 14, 17], "futur": [0, 2, 4, 7, 10, 11, 14], "version": [0, 2, 5, 7, 8, 9, 11, 12, 14], "cufftdx": [0, 2, 3, 5, 6, 10, 12, 13, 14], "mai": [0, 2, 4, 5, 9, 10, 11, 12, 13, 14, 17], "remov": [0, 2, 11, 14], "other": [0, 1, 2, 5, 7, 8, 9, 10, 11, 13, 14], "configur": [0, 2, 4, 5, 8, 9, 14], "do": [0, 2, 9, 11, 13, 14, 17], "continu": [0, 2, 11, 14], "so": [0, 2, 9, 11, 13, 14, 17], "here": [1, 9, 11], "you": [1, 7, 8, 9, 11, 14], "can": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 17], "find": [1, 8], "descript": [1, 2, 5, 6, 9, 10, 11, 14], "main": [1, 7], "compon": [1, 7, 8, 11], "librari": [1, 5, 7, 8, 9, 10, 11, 12, 13, 14], "usag": [1, 10, 11, 13], "exampl": [1, 3, 6, 7, 8, 9, 11, 13], "oper": [1, 2, 4, 5, 7, 9, 10, 11, 12, 14], "execut": [1, 5, 7, 10, 13, 15, 16], "trait": [1, 2, 7, 9, 14], "method": [1, 4, 5, 7, 9, 10, 13], "thread": [1, 5, 6, 9, 10, 12, 14, 15], "block": [1, 5, 6, 9, 10, 12, 13, 14, 15], "make": [1, 4, 5, 9, 10, 11, 14], "function": [1, 3, 4, 7, 9, 10, 11, 12, 13], "These": [2, 11], "ar": [2, 3, 4, 5, 8, 9, 10, 11, 12, 14], "us": [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17], "run": [2, 3, 4, 5, 10, 12, 14], "A": [2, 7, 9, 10, 11], "code": [2, 5, 9, 10, 11, 12], "includ": [2, 3, 4, 5, 8, 9, 10, 11, 13, 14], "hpp": [2, 3, 4, 5, 8, 9, 10], "decltyp": [2, 3, 4, 9, 10], "128": [2, 3, 4, 9, 10], "type": [2, 5, 6, 9, 10, 11, 14, 15], "fft_type": [2, 3, 4, 9, 10], "c2c": [2, 3, 4, 5, 6, 9, 10], "direct": [2, 9, 10, 11, 14, 15], "fft_direct": [2, 3, 4, 9, 10], "forward": [2, 3, 4, 5, 9, 10], "float": [2, 3, 4, 9, 10, 14], "complex_typ": [2, 4, 9, 10], "typenam": [2, 4, 9, 10], "value_typ": [2, 4, 9, 10], "__global__": [2, 9, 10], "kernel": [2, 5, 7, 13], "argument": [2, 9], "pointer": [2, 9, 13], "extern": [2, 9, 10], "__shared__": [2, 9, 10], "shared_mem": [2, 9, 10], "thread_data": [2, 5, 9, 10], "storage_s": [2, 4, 9, 10], "load": [2, 5, 6, 9, 10, 12], "store": [2, 5, 6, 9, 12, 14], "result": [2, 3, 5, 9, 10, 11, 13, 14, 15, 16, 17], "global": [2, 4, 5, 7, 9, 10, 12, 14], "void": [2, 9, 10, 11], "t": [2, 4, 5, 9, 11, 13, 14, 17], "defin": [2, 3, 4, 5, 7, 14], "descriptor": [2, 3, 4, 9, 10, 14], "ani": [2, 4, 10, 11], "float2": 2, "double2": 2, "long": [2, 5], "its": [2, 4, 5, 9, 11], "align": [2, 5], "element": [2, 9, 10, 12, 14, 15], "same": [2, 3, 4, 5, 9, 15, 16], "those": [2, 11], "thi": [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14], "avail": [2, 4, 8, 10, 11, 12, 13, 14], "ha": [2, 4, 5, 9, 10, 11], "been": [2, 4, 10, 11], "construct": [2, 3, 4, 10, 11], "is_complete_fft_execut": [2, 4, 10], "true": [2, 3, 4, 8], "arrai": [2, 5, 9], "should": [2, 3, 4, 9, 10, 12, 13], "per": [2, 5, 9, 10, 12, 14, 15], "must": [2, 3, 4, 9, 11, 14], "fit": [2, 11], "It": [2, 3, 5, 9, 15, 16], "guarante": [2, 3, 14, 15, 16], "exactli": [2, 3, 16], "gpu": [2, 3, 5, 9, 10, 11, 12, 14, 16], "differ": [2, 3, 4, 5, 7, 9, 10, 11, 15, 16], "cuda": [2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17], "architectur": [2, 3, 4, 9, 10, 13, 14, 16], "produc": [2, 3, 15, 16], "bit": [2, 3, 13, 15, 16], "ident": [2, 3, 12, 15, 16], "1": [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14], "shared_memori": 2, "workspace_typ": [2, 4, 9, 10], "which": [2, 3, 4, 5, 9, 10, 11, 13, 14], "don": [2, 9, 11], "shared_memory_input": 2, "4": [2, 4, 5, 11, 12], "alignof": 2, "when": [2, 3, 4, 8, 9, 13], "requires_workspac": [2, 3, 4, 5, 9, 14], "fals": [2, 3, 4], "overload": 2, "otherwis": [2, 4, 11], "user": [2, 3, 4, 5, 8, 9, 10, 11, 12, 14], "pass": [2, 4, 5, 8, 9, 13, 14, 17], "refer": [2, 7], "shared_memory_s": [2, 4, 9, 10], "byte": [2, 4, 9, 10], "The": [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14], "place": 2, "mean": [2, 4, 5, 10, 11], "back": [2, 5, 9], "an": [2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 17], "addit": [2, 9, 11, 14], "commun": [2, 11], "between": [2, 4, 5, 11], "thu": [2, 8], "all": [2, 4, 5, 8, 9, 11, 14], "smaller": [2, 3], "than": [2, 3, 4, 10], "e": [2, 3], "maximum": [2, 4, 5, 10, 11], "ffts_per_block": [2, 3, 4, 9, 10], "fft_input_size_in_byt": 2, "fft_output_size_in_byt": 2, "number": [2, 3, 4, 7, 8, 9, 10, 11, 15], "elementsperthread": [2, 3, 4, 9, 10, 13, 15], "calcul": [2, 3, 4, 5, 7, 9, 10, 12, 15], "fftsperblock": [2, 3, 4, 9, 10, 13, 15], "dimens": [2, 3, 4, 5, 14, 15], "blockdim": [2, 14, 15], "For": [2, 3, 4, 5, 9, 10, 11, 12], "complex": [2, 3, 4, 5, 6, 9, 12, 14], "singl": [2, 3, 5, 9, 10, 14], "doubl": [2, 3, 4, 9, 14], "first": [2, 5, 7, 13], "real": [2, 3, 5, 6, 9, 10, 14], "part": [2, 4, 8, 11], "second": [2, 9], "imaginari": 2, "process": [2, 5, 9, 12], "fp16": [2, 5], "implicitli": [2, 5], "comput": [2, 3, 4, 9, 10, 11, 12, 14], "two": [2, 3, 5, 10, 13], "expect": [2, 4], "order": [2, 7, 9, 12, 13, 14, 17], "real_1": 2, "real_2": 2, "imaginary_1": 2, "imaginary_2": 2, "r2c": [2, 3, 4, 5, 6], "c2r": [2, 3, 4, 5, 6], "logic": 2, "each": [2, 3, 4, 5, 9, 11], "contain": [2, 5, 11], "see": [2, 3, 4, 5, 9, 13, 14], "also": [2, 4, 5, 9, 11], "implicit_type_batch": [2, 4], "section": [2, 4, 5, 8, 9, 10, 11, 13], "describ": [2, 3, 5, 9, 10, 11], "n": 2, "th": [2, 4], "index": [2, 9, 10, 12], "from": [2, 4, 5, 7, 9, 10, 11, 12, 14], "0": [2, 4, 5, 7, 8, 9, 10, 14], "particip": [2, 11], "stride": [2, 9, 10, 13], "where": [2, 11, 14], "later": 2, "rule": 2, "8": [2, 3, 4, 9, 11, 12], "point": [2, 3, 4, 9], "equal": [2, 4, 9, 10], "have": [2, 4, 5, 8, 9, 10, 11], "natur": [2, 9, 11], "": [2, 3, 4, 5, 9, 11], "import": [2, 9], "note": [2, 4, 5, 7], "larg": 2, "more": [2, 3, 4, 9, 10, 12, 14], "48": 2, "kb": 2, "therefor": [2, 9, 14], "program": [2, 9, 11, 13], "guid": [2, 5, 9, 12, 13], "dynam": [2, 10], "rather": 2, "static": [2, 10], "addition": [2, 11], "explicit": 2, "opt": [2, 10], "cudafuncsetattribut": [2, 10], "set": [2, 3, 4, 5, 8, 9, 10, 11, 12], "cudafuncattributemaxdynamicsharedmemorys": [2, 10], "below": [2, 11, 12, 14], "introduct": [2, 6, 7, 9, 13], "namespac": [2, 9, 10], "16384": [2, 3, 14], "sm": [2, 4, 9, 10, 14], "800": [2, 3, 4], "block_fft_kernel": [2, 9, 10], "increas": [2, 10, 12], "max": [2, 10, 14], "match": [2, 5, 10, 12], "invok": [2, 9, 10], "block_dim": [2, 3, 4, 9, 10], "templat": [2, 4, 9, 10, 13], "class": [2, 4, 9, 10, 14, 17], "auto": [2, 4, 9, 10], "make_workspac": [2, 4, 9, 10], "cudaerror_t": [2, 4, 9, 10], "error": [2, 4, 9, 11, 14, 17], "helper": [2, 4, 7], "creat": [2, 3, 4, 5, 9, 11, 14], "If": [2, 3, 4, 9, 11], "after": [2, 8, 9, 10], "call": [2, 5, 10, 13], "cudasuccess": [2, 9, 10], "wa": [2, 4, 8, 10], "correctli": 2, "invalid": [2, 11], "doesn": [2, 4, 5], "empti": [2, 4], "alloc": [2, 4, 5, 9, 10, 14], "object": [2, 3, 4, 9, 10], "valid": [2, 4], "onli": [2, 3, 4, 5, 8, 9, 10, 11, 14], "howev": [2, 10, 14], "never": 2, "workspace_s": [2, 4], "respons": [2, 11], "free": 2, "concurr": [2, 3], "sinc": [2, 5, 9, 14], "copi": [2, 5, 11], "underli": [2, 3, 4], "race": 2, "freed": 2, "upon": [2, 11], "destruct": 2, "last": [2, 9, 13], "cast": [2, 4], "track": [2, 4], "lifetim": [2, 4], "within": [2, 3, 4, 11], "return": [2, 4, 9], "__launch_bounds__": 2, "max_threads_per_block": [2, 4], "unsign": [3, 4, 9, 10], "int": [3, 4, 9, 10], "p": 3, "cc": 3, "f": 3, "x": [3, 4, 8, 9, 10], "y": [3, 4, 8, 9, 10], "z": [3, 4, 8], "solv": 3, "thei": [3, 4, 5, 8, 11], "divid": [3, 4], "default": [3, 4, 9, 10, 12, 13], "valu": [3, 5, 9, 10, 12, 13], "Not": 3, "either": [3, 4, 5, 10, 11], "invers": [3, 4, 5, 9], "input": [3, 7, 9, 10, 11, 14], "output": [3, 7, 9, 10, 11, 14], "data": [3, 4, 5, 6, 7, 9, 10, 11, 12, 14], "__half": [3, 4], "target": [3, 4, 8, 9, 10, 12, 13, 14], "architecur": 3, "gener": [3, 5, 7, 9, 10, 11, 13], "combin": [3, 9, 14], "form": [3, 4, 9, 11], "complet": [3, 5, 10], "ad": [3, 4, 9, 10, 13, 14], "consist": [3, 7, 10, 11], "One": [3, 4], "one": [3, 4, 5, 9, 10, 12, 14], "unless": [3, 4, 11, 13], "There": [3, 4], "restrict": [3, 11], "greater": [3, 4, 14], "assum": [3, 9], "necessari": [3, 4, 5, 11], "specifi": [3, 9, 11], "cuffdx": [3, 10], "perform": [3, 4, 6, 7, 9, 10, 11, 13, 14], "unnorm": 3, "fast": [3, 7], "fourier": [3, 7], "transform": [3, 7], "well": [3, 5, 9], "support": [3, 4, 5, 7, 9, 10, 11, 13], "volta": [3, 9, 12], "700": [3, 4, 9, 10], "720": 3, "sm_70": [3, 9], "sm_72": 3, "ture": [3, 12], "750": [3, 4], "sm_75": 3, "amper": [3, 12], "860": 3, "870": 3, "sm_80": 3, "sm_86": 3, "sm_87": 3, "ada": [3, 13], "890": 3, "sm_89": 3, "hopper": [3, 12, 13], "900": 3, "sm_90": 3, "how": [3, 4, 5, 8, 9, 10, 12], "simultan": 3, "context": [3, 4], "independ": [3, 11], "mutual": 3, "exclus": [3, 11], "compil": [3, 4, 5, 7, 8, 10, 13, 17], "fail": [3, 4, 11, 14, 17], "With": 3, "rang": [3, 5, 9, 10, 14], "32": [3, 13, 14], "16": [3, 14], "collect": [3, 4, 9, 14], "cooper": [3, 11], "insid": [3, 7, 9], "2d": 3, "size_of": [3, 4, 9, 10], "elements_per_thread": [3, 4, 9, 10], "implement": [3, 4, 5, 9, 10], "yet": [3, 14], "u": [3, 11], "heurist": [3, 4, 10], "custom": [3, 5, 7, 9, 12, 13, 14], "allow": [3, 9, 10, 11], "tune": [3, 12], "parallel": [3, 4, 9, 12], "separ": [3, 5, 11, 12], "group": [3, 5, 6], "determin": [3, 4, 9, 11, 12], "divisor": 3, "request": [3, 9, 11], "struct": [3, 4], "3d": 3, "type_of": 4, "direction_of": 4, "precision_of": 4, "is_fft": 4, "is_fft_execut": 4, "is_complete_fft": 4, "input_typ": 4, "output_typ": 4, "suggested_ffts_per_block": [4, 10], "provid": [4, 5, 7, 8, 9, 11, 12], "inform": [4, 9, 11, 13, 14], "about": [4, 9, 13, 14], "three": [4, 7, 9], "categori": 4, "none": 4, "retriev": [4, 10], "iostream": 4, "is_complet": 4, "std": [4, 5, 6, 8, 9], "cout": 4, "endl": 4, "messag": 4, "miss": 4, "abl": 4, "both": [4, 5, 9, 11], "we": [4, 9, 10, 12, 13], "directli": 4, "depend": [4, 5, 9, 10, 12, 14], "build": [4, 9], "detail": [4, 9, 10, 14], "alwai": [4, 12], "held": 4, "built": 4, "header": [4, 5, 8, 9, 10, 11, 14], "file": [4, 5, 8, 9, 11], "higher": [4, 14], "multipl": [4, 5, 10, 11], "go": [4, 8, 9, 10], "pleas": [4, 5, 9, 11, 13], "releas": [4, 7, 8, 10, 11], "replac": 4, "extend": [4, 11], "devic": [4, 5, 7, 9], "side": 4, "dim3": [4, 14, 17], "total": [4, 11], "extra": [4, 6, 7, 12], "manag": [4, 7, 9, 10, 11], "size_byt": [4, 9, 10], "sizeof": [4, 9, 10], "cudamallocmanag": [4, 9, 10], "check": [4, 8, 9, 14], "_valuetyp": 4, "label": [4, 7], "format": [4, 5, 8, 9], "maxim": [4, 9], "paramet": [4, 7, 9, 12, 14], "boolean": 4, "doe": [4, 9, 13], "won": 4, "much": 4, "zero": 4, "without": [4, 11, 12], "integ": 4, "repres": [4, 11], "bool_const": 4, "variabl": [4, 14, 17], "inlin": [4, 10], "constexpr": [4, 14, 17], "bool": 4, "is_supported_v": 4, "whether": [4, 11, 14], "deduc": 4, "via": [4, 10, 11, 14], "take": [4, 9, 11, 12], "account": 4, "verifi": [4, 9, 11], "8192": [4, 14], "4095": 4, "level": [5, 10], "sampl": [5, 9, 11], "cover": 5, "precis": [5, 6, 7, 9, 10, 14, 15], "few": 5, "special": [5, 11], "highlight": 5, "benefit": 5, "subgroup": [5, 6], "introduction_exampl": [5, 6, 9, 10], "api": [5, 6, 7, 9, 10, 11, 12, 13], "simple_fft_thread_fp16": [5, 6], "half": [5, 6, 14], "simple_fft_block_r2c": [5, 6], "simple_fft_block_c2r": [5, 6], "simple_fft_block_half2": [5, 6], "__half2": [5, 6], "simple_fft_block_fp16": [5, 6], "simple_fft_block_r2c_fp16": [5, 6], "simple_fft_block_c2r_fp16": [5, 6], "simple_fft_block_shar": [5, 6], "share": [5, 6, 7, 9, 12, 13, 14], "memori": [5, 6, 7, 9, 13, 14], "simple_fft_block_std_complex": [5, 6], "simple_fft_block_cub_io": [5, 6], "cub": [5, 6], "nvrtc_fft_thread": [5, 6], "nvrtc_fft_block": [5, 6], "block_fft_perform": [5, 6], "benchmark": [5, 6], "block_fft_performance_mani": [5, 6], "simplifi": [5, 6, 8], "convolution_r2c_c2r": [5, 6], "convolution_perform": [5, 6], "cufft": [5, 6, 7, 9, 12], "document": [5, 7, 11, 13], "explain": 5, "basic": [5, 7], "introductori": 5, "list": [5, 8, 9, 12, 13, 14], "abov": [5, 9], "dimension": 5, "routin": [5, 7, 10], "host": [5, 9, 10, 13, 14, 17], "buffer": 5, "final": [5, 9], "showcas": 5, "implicit": 5, "batch": [5, 7, 9, 10], "ie": [5, 12], "show": 5, "correct": [5, 9, 14], "amount": [5, 9], "regist": [5, 9, 10, 12, 14], "them": [5, 10], "transfer": [5, 11], "block_io": 5, "simple_fft_block_": 5, "_fp16": 5, "instead": 5, "reason": [5, 11, 12], "accordingli": 5, "rearrang": 5, "introduc": [5, 11], "becaus": [5, 10], "work": [5, 11, 13, 14], "nvidia": [5, 8, 11, 12], "http": [5, 8, 12], "github": 5, "com": [5, 8, 11, 12], "13": 5, "newer": [5, 14], "present": [5, 9, 14], "runtim": [5, 12], "experiment": [5, 9, 14], "report": [5, 14, 17], "easili": 5, "modifi": [5, 10, 11], "test": [5, 11], "particular": [5, 10, 11], "want": [5, 9, 11], "problem": [5, 9], "chang": [5, 10, 11, 12, 13], "option": [5, 7, 8, 11, 12, 13, 14, 17], "path": [5, 8], "pointwis": 5, "callback": 5, "cufftdx_examples_cufft_callback": 5, "cmake": [5, 7, 14], "ON": [5, 11], "dcufftdx_examples_cufft_callback": 5, "given": [5, 14], "improv": [5, 7, 12, 13], "20": 5, "3x": 5, "speed": 5, "a100": [5, 13], "80gb": 5, "fig": 5, "comparison": 5, "scale": 5, "util": [5, 11], "clock": 5, "chart": 5, "rel": [5, 11, 14], "compar": 5, "light": 5, "blue": 5, "accord": [5, 11], "layout": 5, "o": [5, 8, 9], "promis": 5, "deliv": [5, 11], "best": [5, 12], "everi": [5, 14], "write": [5, 11, 12], "own": [5, 11], "need": [5, 7, 8, 9, 10, 12, 14], "simpl": [6, 7, 10, 12], "simple_fft_thread": 6, "simple_fft_block": 6, "nvrtc": [6, 7, 9, 14], "convolut": [6, 7], "extens": [7, 10], "enabl": [7, 12, 13, 14, 17], "your": [7, 9, 11, 13], "fuse": [7, 9], "decreas": 7, "latenc": 7, "applic": [7, 11, 12], "featur": [7, 10, 11], "quick": [7, 9, 13], "start": [7, 12], "comprehens": 7, "overview": 7, "embedd": 7, "high": 7, "unnecessari": 7, "movement": 7, "customiz": 7, "adjust": [7, 12], "select": [7, 9, 10, 14], "etc": [7, 9, 10], "abil": [7, 12], "save": [7, 9, 10, 12], "trip": [7, 12], "compat": [7, 10, 11], "toolkit": [7, 10, 14], "instal": [7, 9, 11, 13], "project": [7, 9, 11], "launch": [7, 10, 12], "next": [7, 13], "optim": [7, 12, 13], "what": [7, 9, 12], "happen": 7, "under": [7, 11], "hood": 7, "achiev": 7, "advic": 7, "fusion": 7, "advanc": 7, "further": 7, "read": 7, "softwar": [7, 14], "licens": 7, "agreement": 7, "distribut": [8, 11], "mathdx": 8, "packag": 8, "To": [8, 9, 10], "download": [8, 11], "most": [8, 14], "recent": 8, "develop": [8, 11], "websit": 8, "just": 8, "directori": [8, 9], "command": 8, "nvcc": [8, 9, 13, 14, 17], "c": [8, 9, 10, 12, 14], "17": [8, 9, 14], "arch": [8, 9], "sm_xy": 8, "cufftdx_include_directori": 8, "your_source_fil": 8, "cu": [8, 9], "your_binari": 8, "unpack": 8, "yy": 8, "mm": 8, "tarbal": 8, "your_directori": 8, "locat": [8, 9], "review": 8, "makefil": 8, "ship": [8, 9, 10, 11], "alongsid": 8, "find_packag": 8, "link": 8, "propag": [8, 9], "cufftdx_include_dir": 8, "config": 8, "target_link_librari": 8, "yourprogram": 8, "altern": 8, "mathdx_root": 8, "dure": [8, 14], "dmathdx_root": 8, "cufftdx_found": 8, "mathdx_cufftdx_found": 8, "found": [8, 9], "mathdx_vers": 8, "cufftdx_vers": 8, "matrix": 8, "22": 8, "02": 8, "standalon": 9, "base": [9, 10, 11], "step": [9, 11], "done": [9, 12], "togeth": 9, "evalu": 9, "time": [9, 10, 11], "encod": 9, "properti": [9, 11], "As": [9, 10, 11], "mention": 9, "befor": 9, "obtain": [9, 10, 14], "fulli": [9, 10, 11], "usabl": 9, "piec": 9, "mani": 9, "would": [9, 11], "like": [9, 10], "map": 9, "mode": [9, 10], "exact": 9, "influenc": 9, "add": 9, "indic": [9, 10, 11], "choic": [9, 10, 11], "potenti": [9, 12], "affect": [9, 11], "re": 9, "onc": 9, "ask": [9, 11], "being": 9, "sai": 9, "instanti": 9, "cost": [9, 11], "seen": 9, "handl": [9, 10], "queri": [9, 10], "resourc": [9, 12], "some": [9, 10, 11], "know": 9, "fix": [9, 11], "manner": [9, 10, 11], "cuda_check_and_exit": 9, "marco": 9, "print": 9, "exit": 9, "error_cod": [9, 10], "cudapeekatlasterror": 9, "cudadevicesynchron": [9, 10], "equival": [9, 11], "similarli": 9, "simplic": 9, "full": [9, 10, 11, 13], "local": [9, 10], "id": [9, 10], "const": [9, 10], "local_fft_id": [9, 10], "threadidx": [9, 10], "grid": [9, 10], "plu": [9, 10], "global_fft_id": [9, 10], "blockidx": [9, 10], "offset": [9, 10], "sure": [9, 10], "out": [9, 10, 11], "bound": [9, 10, 12], "size_t": [9, 10], "cudafre": [9, 10], "notic": [9, 10, 11], "unlik": 9, "move": 9, "major": [9, 11], "advantag": 9, "pre": [9, 11, 12], "post": [9, 12], "o3": 9, "path_to_cufftdx_loc": 9, "world": 10, "case": [10, 11, 12, 13], "aim": 10, "design": [10, 11], "burden": 10, "automat": 10, "while": [10, 11], "offer": [10, 12, 13], "control": [10, 11], "over": [10, 11], "defer": 10, "definit": 10, "certain": 10, "suggest": [10, 11, 14], "itself": 10, "manual": 10, "field": 10, "fft_base": 10, "lack": 10, "larger": 10, "attribut": 10, "express": [10, 11], "variat": 10, "techniqu": 10, "mechan": [10, 11], "attach": 10, "expos": 10, "ptx": 10, "abstract": 10, "proof": 10, "By": [10, 11], "exist": [10, 12], "new": [10, 11], "On": 10, "platform": 10, "adapt": 10, "quickli": 10, "evolv": 10, "hardwar": 10, "approach": 10, "wai": 10, "hand": 10, "sourc": [10, 11, 12], "decoupl": 10, "abi": 10, "along": 10, "driver": 10, "recompil": 10, "organ": [10, 11], "preserv": 10, "get": 10, "pick": 10, "FOR": 11, "math": 11, "kit": 11, "legal": 11, "corpor": 11, "govern": 11, "discret": 11, "sdk": 11, "materi": 11, "item": 11, "asset": 11, "imag": 11, "textur": 11, "model": 11, "scene": 11, "video": 11, "nativ": 11, "binari": 11, "accept": 11, "adult": 11, "ag": 11, "countri": 11, "enter": 11, "behalf": 11, "compani": 11, "entiti": 11, "author": 11, "bind": 11, "term": 11, "condit": 11, "agre": 11, "purpos": 11, "permit": 11, "b": 11, "law": 11, "regul": 11, "practic": [11, 12], "guidelin": 11, "relev": 11, "jurisdict": 11, "grant": 11, "subject": 11, "herebi": 11, "non": 11, "right": 11, "sublicens": 11, "except": 11, "expressli": 11, "identifi": 11, "incorpor": 11, "meet": 11, "exercis": 11, "beyond": 11, "portion": 11, "shall": 11, "access": [11, 13], "modif": 11, "deriv": 11, "tool": 11, "intern": 11, "limit": [11, 13], "relat": 11, "protect": 11, "intellectu": 11, "privaci": 11, "secur": 11, "notifi": 11, "known": 11, "suspect": 11, "complianc": 11, "enforc": 11, "respect": 11, "employe": 11, "contractor": 11, "subsidiari": 11, "network": 11, "academ": 11, "institut": 11, "enrol": 11, "emploi": 11, "becom": 11, "awar": [11, 13], "didn": 11, "resolv": 11, "prevent": 11, "occurr": 11, "alpha": 11, "beta": 11, "preview": 11, "flaw": 11, "reduc": 11, "reliabl": 11, "standard": [11, 14], "commerci": 11, "unexpect": 11, "loss": 11, "delai": 11, "unpredict": 11, "damag": 11, "risk": 11, "understand": 11, "intend": 11, "product": 11, "busi": 11, "critic": 11, "system": 11, "choos": 11, "abandon": 11, "termin": 11, "liabil": 11, "updat": [11, 13], "patch": 11, "workaround": 11, "deem": 11, "content": 11, "prior": 11, "maintain": 11, "incompat": 11, "third": 11, "parti": 11, "proprietari": 11, "accompani": 11, "open": 11, "extent": 11, "conflict": [11, 13], "associ": 11, "reserv": 11, "titl": 11, "interest": 11, "appli": 11, "revers": 11, "engin": 11, "decompil": 11, "disassembl": 11, "copyright": 11, "sell": 11, "rent": 11, "sponsor": 11, "endors": 11, "bypass": 11, "disabl": 11, "circumv": 11, "encrypt": 11, "digit": 11, "authent": 11, "caus": 11, "disclos": 11, "ii": 11, "iii": 11, "redistribut": 11, "charg": 11, "acknowledg": 11, "certifi": 11, "connect": 11, "mainten": 11, "failur": 11, "could": 11, "situat": 11, "threaten": 11, "safeti": 11, "human": 11, "life": 11, "catastroph": 11, "avion": 11, "navig": 11, "autonom": 11, "vehicl": 11, "ai": 11, "solut": 11, "automot": 11, "militari": 11, "medic": 11, "liabl": 11, "whole": 11, "claim": 11, "aris": 11, "sole": 11, "ensur": [11, 12], "servic": 11, "suffici": 11, "compli": 11, "regulatori": 11, "defend": 11, "indemnifi": 11, "hold": 11, "harmless": 11, "affili": 11, "agent": 11, "offic": 11, "director": 11, "against": 11, "oblig": 11, "debt": 11, "fine": [11, 14], "restitut": 11, "expens": 11, "attornei": 11, "fee": 11, "incid": 11, "establish": 11, "indemnif": 11, "outsid": 11, "scope": 11, "ownership": 11, "licensor": 11, "beneficiari": 11, "feedback": 11, "regard": 11, "possibl": 11, "enhanc": 11, "voluntarili": 11, "perpetu": 11, "worldwid": 11, "irrevoc": 11, "reproduc": 11, "through": 11, "tier": 11, "sublicense": 11, "distributor": 11, "payment": 11, "royalti": 11, "No": 11, "warranti": 11, "THE": 11, "BY": 11, "AS": 11, "AND": 11, "WITH": 11, "fault": 11, "TO": 11, "ITS": 11, "disclaim": 11, "OF": 11, "kind": 11, "OR": 11, "impli": 11, "statutori": 11, "BUT": 11, "NOT": 11, "merchant": 11, "infring": 11, "absenc": 11, "defect": 11, "therein": 11, "latent": 11, "patent": 11, "NO": 11, "made": 11, "basi": 11, "trade": 11, "cours": 11, "deal": 11, "BE": 11, "incident": 11, "punit": 11, "consequenti": 11, "lost": 11, "profit": 11, "goodwil": 11, "procur": 11, "substitut": 11, "IN": 11, "SUCH": 11, "breach": 11, "contract": 11, "tort": 11, "neglig": 11, "action": 11, "theori": 11, "event": 11, "WILL": 11, "cumul": 11, "exce": 11, "00": 11, "suit": 11, "enlarg": 11, "regardless": 11, "advis": 11, "remedi": 11, "essenti": 11, "bargain": 11, "absent": 11, "provis": 11, "econom": 11, "substanti": 11, "until": 11, "stop": 11, "thirti": 11, "30": 11, "dai": 11, "immedi": 11, "violat": 11, "commenc": 11, "proceed": 11, "decid": 11, "longer": 11, "viabl": 11, "promptli": 11, "discontinu": 11, "destroi": 11, "possess": 11, "written": 11, "commit": 11, "surviv": 11, "wish": 11, "assign": 11, "merger": 11, "consolid": 11, "dissolut": 11, "contact": 11, "permiss": 11, "attempt": 11, "approv": 11, "effect": 11, "deleg": 11, "unit": 11, "state": 11, "delawar": 11, "entir": [11, 12], "resid": 11, "principl": 11, "nation": 11, "convent": 11, "sale": 11, "good": 11, "specif": [11, 13, 14], "english": 11, "languag": 11, "feder": 11, "court": 11, "santa": 11, "clara": 11, "counti": 11, "california": 11, "disput": 11, "notwithstand": 11, "still": 11, "injunct": 11, "urgent": 11, "relief": 11, "compet": 11, "illeg": 11, "unenforc": 11, "constru": 11, "remain": 11, "forc": 11, "privat": 11, "duplic": 11, "disclosur": 11, "subcontractor": 11, "pursuant": 11, "dfar": 11, "227": 11, "7202": 11, "forth": 11, "subparagraph": 11, "claus": 11, "far": 11, "52": 11, "19": 11, "manufactur": 11, "2788": 11, "san": 11, "toma": 11, "expresswai": 11, "ca": 11, "95051": 11, "export": 11, "prohibit": 11, "bureau": 11, "industri": 11, "sanction": 11, "administ": 11, "depart": 11, "treasuri": 11, "foreign": 11, "ofac": 11, "destin": 11, "end": 11, "confirm": 11, "citizen": 11, "current": 11, "embargo": 11, "receiv": 11, "mail": 11, "email": 11, "fax": 11, "send": 11, "electron": 11, "satisfi": 11, "correspond": 11, "america": 11, "attent": 11, "constitut": 11, "matter": 11, "supersed": 11, "negoti": 11, "exchang": 11, "issu": 11, "null": 11, "amend": 11, "waiver": 11, "sign": 11, "suitabl": 11, "question": 11, "v": 11, "februari": 11, "2022": 11, "better": 12, "greatli": 12, "regular": 12, "baselin": 12, "magnitud": 12, "wors": 12, "port": 12, "analysi": 12, "help": 12, "try": 12, "might": [12, 14], "enough": 12, "fill": 12, "peak": 12, "merg": 12, "adjac": 12, "avoid": [12, 13], "coalesc": 12, "temporari": 12, "consid": 12, "tweak": 12, "upcom": 12, "stream": 12, "occup": 12, "cudaoccupancymaxactiveblockspermultiprocessor": 12, "optimum": 12, "nsight": 12, "lose": 12, "doc": 12, "html": 12, "nsightcomput": 12, "group__cudart__occup": 12, "signific": 13, "variou": 13, "impact": 13, "around": 13, "initi": 13, "orin": 13, "sm87": 13, "sm89": 13, "sm90": 13, "is_support": [13, 14], "preliminari": [13, 14], "msvc": [13, 14, 17], "chapter": 13, "__cplusplu": [13, 14, 17], "zc": [13, 14, 17], "xcompil": [13, 14, 17], "phase": 13, "length": 13, "mangl": 13, "name": 13, "extrem": 13, "incorrect": 13, "instanc": 13, "involv": 13, "ga": 13, "sm80": 13, "sm70": 13, "v100": 13, "restor": 13, "ptxa": 13, "warn": 13, "address": 13, "line": 13, "xxx": 13, "address_s": 13, "64": 13, "shouldn": 13, "appear": 13, "anymor": 13, "earli": 13, "ea": 13, "small": 14, "18": 14, "194": 14, "gcc": 14, "clang": 14, "9": 14, "linux": 14, "wsl2": 14, "1920": 14, "window": 14, "visual": 14, "studio": 14, "2019": 14, "declar": [14, 17], "emit": 14, "unsupport": 14, "silenc": 14, "cufftdx_ignore_deprecated_compil": 14, "cufttdx": 14, "cufftdx_ignore_deprecated_dialect": 14, "collabor": 14, "expertis": 14, "bi": 14, "flow": 14, "max_siz": 14, "tabl": 14, "max_size_fp64": 14, "summar": 14, "75": 14, "4096": 14, "70": 14, "72": 14, "86": 14, "89": 14, "80": 14, "87": 14, "90": 14, "2048": 14, "grain": 14}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"cufftdx": [1, 4, 7, 8, 9], "api": 1, "refer": [1, 12], "execut": [2, 3, 4, 9], "method": 2, "thread": [2, 3, 4], "block": [2, 3, 4], "valu": [2, 4], "format": 2, "half": 2, "precis": [2, 3, 4], "implicit": [2, 4], "batch": [2, 4], "input": [2, 4, 5], "output": [2, 4, 5], "data": 2, "In": [2, 8], "regist": 2, "exampl": [2, 4, 5], "share": [2, 4, 10], "memori": [2, 4, 10, 12], "usag": 2, "make": 2, "workspac": [2, 4], "function": [2, 5, 14], "oper": 3, "descript": [3, 4], "size": [3, 4], "direct": [3, 4], "type": [3, 4], "sm": 3, "configur": 3, "fft": [3, 4, 5, 9, 10], "per": [3, 4], "element": [3, 4], "blockdim": 3, "trait": 4, "i": 4, "complet": 4, "storag": 4, "stride": 4, "suggest": 4, "dim": 4, "max": 4, "requir": [4, 14], "other": 4, "is_support": 4, "introduct": 5, "simpl": 5, "simple_fft_thread": 5, "simple_fft_block": 5, "extra": [5, 10], "nvrtc": 5, "perform": [5, 12], "convolut": 5, "helper": 5, "nvidia": 7, "highlight": 7, "user": 7, "guid": [7, 8], "quick": 8, "instal": 8, "your": [8, 10], "project": 8, "cmake": 8, "defin": [8, 9], "variabl": 8, "first": 9, "us": [9, 10], "basic": 9, "launch": 9, "kernel": [9, 10, 12], "compil": [9, 14], "next": 10, "custom": 10, "optim": 10, "paramet": 10, "what": 10, "happen": 10, "under": 10, "hood": 10, "why": 10, "softwar": 11, "licens": 11, "agreement": 11, "achiev": 12, "high": 12, "gener": 12, "advic": 12, "manag": 12, "fusion": 12, "advanc": 12, "further": 12, "read": 12, "releas": 13, "note": 13, "1": 13, "0": 13, "new": 13, "featur": 13, "known": 13, "issu": 13, "resolv": 13, "3": 13, "support": 14}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"cuFFTDx API Reference": [[1, "cufftdx-api-reference"]], "Execution Methods": [[2, "execution-methods"]], "Thread Execute Method": [[2, "thread-execute-method"]], "Block Execute Method": [[2, "block-execute-method"]], "Value Format": [[2, "value-format"]], "Half Precision Implicit Batching": [[2, "half-precision-implicit-batching"]], "Input/Output Data Format": [[2, "input-output-data-format"]], "Data In Registers": [[2, "data-in-registers"]], "Example": [[2, null], [4, null], [4, null]], "Data In Shared Memory": [[2, "data-in-shared-memory"]], "Shared Memory Usage": [[2, "shared-memory-usage"]], "Make Workspace Function": [[2, "make-workspace-function"]], "Operators": [[3, "operators"]], "Description Operators": [[3, "description-operators"]], "Size Operator": [[3, "size-operator"]], "Direction Operator": [[3, "direction-operator"]], "Type Operator": [[3, "type-operator"]], "Precision Operator": [[3, "precision-operator"]], "SM Operator": [[3, "sm-operator"]], "Execution Operators": [[3, "execution-operators"]], "Thread Operator": [[3, "thread-operator"]], "Block Operator": [[3, "block-operator"]], "Block Configuration Operators": [[3, "block-configuration-operators"]], "FFTs Per Block Operator": [[3, "ffts-per-block-operator"]], "Elements Per Thread Operator": [[3, "elements-per-thread-operator"]], "BlockDim Operator": [[3, "blockdim-operator"]], "Traits": [[4, "traits"]], "Description Traits": [[4, "description-traits"]], "Size Trait": [[4, "size-trait"]], "Type Trait": [[4, "type-trait"]], "Direction Trait": [[4, "direction-trait"]], "Precision Trait": [[4, "precision-trait"]], "Is FFT? Trait": [[4, "is-fft-trait"]], "Is FFT Execution? Trait": [[4, "is-fft-execution-trait"]], "Is FFT-complete? Trait": [[4, "is-fft-complete-trait"]], "Is FFT-complete Execution? Trait": [[4, "is-fft-complete-execution-trait"]], "Execution Traits": [[4, "execution-traits"]], "Thread Traits": [[4, "thread-traits"]], "Value Type Trait": [[4, "value-type-trait"], [4, "valuetype-block-trait-label"]], "Input Type Trait": [[4, "input-type-trait"], [4, "inputtype-block-trait-label"]], "Output Type Trait": [[4, "output-type-trait"], [4, "outputtype-block-trait-label"]], "Implicit Type Batching Trait": [[4, "implicit-type-batching-trait"], [4, "implicit-type-batching-block-trait-label"]], "Elements Per Thread Trait": [[4, "elements-per-thread-trait"], [4, "ept-block-trait-label"]], "Storage Size Trait": [[4, "storage-size-trait"], [4, "storage-block-trait-label"]], "Stride Size Trait": [[4, "stride-size-trait"], [4, "stride-block-trait-label"]], "Block Traits": [[4, "block-traits"]], "Workspace Type Trait": [[4, "workspace-type-trait"]], "FFTs Per Block Trait": [[4, "ffts-per-block-trait"]], "Suggested FFTs Per Block Trait": [[4, "suggested-ffts-per-block-trait"]], "Shared Memory Size Trait": [[4, "shared-memory-size-trait"]], "Block Dim Trait": [[4, "block-dim-trait"]], "Max Threads Per Block Trait": [[4, "max-threads-per-block-trait"]], "Requires Workspace Trait": [[4, "requires-workspace-trait"]], "Workspace Size Trait": [[4, "workspace-size-trait"]], "Other Traits": [[4, "other-traits"]], "cufftdx::is_supported": [[4, "cufftdx-is-supported"]], "Examples": [[5, "examples"]], "Introduction Examples": [[5, "introduction-examples"]], "Simple FFT Examples": [[5, "simple-fft-examples"]], "simple_fft_thread* Examples": [[5, "simple-fft-thread-examples"]], "simple_fft_block* Examples": [[5, "simple-fft-block-examples"]], "Extra simple_fft_block(*) Examples": [[5, "extra-simple-fft-block-examples"]], "NVRTC Examples": [[5, "nvrtc-examples"]], "FFT Performance": [[5, "fft-performance"]], "Convolution Examples": [[5, "convolution-examples"]], "Input/Output Helper Functions": [[5, "input-output-helper-functions"]], "NVIDIA cuFFTDx": [[7, "nvidia-cufftdx"]], "Highlights": [[7, "highlights"]], "User guide:": [[7, null]], "Quick Installation Guide": [[8, "quick-installation-guide"]], "cuFFTDx In Your Project": [[8, "cufftdx-in-your-project"]], "cuFFTDx In Your CMake Project": [[8, "cufftdx-in-your-cmake-project"]], "Defined Variables": [[8, "defined-variables"]], "First FFT Using cuFFTDx": [[9, "first-fft-using-cufftdx"]], "Defining Basic FFT": [[9, "defining-basic-fft"]], "Executing FFT": [[9, "executing-fft"]], "Launching FFT Kernel": [[9, "launching-fft-kernel"]], "Compilation": [[9, "compilation"]], "Your Next Custom FFT Kernels": [[10, "your-next-custom-fft-kernels"]], "Using Optimal parameters": [[10, "using-optimal-parameters"]], "Extra Shared Memory": [[10, "extra-shared-memory"]], "What happens under the hood": [[10, "what-happens-under-the-hood"]], "Why?": [[10, "why"]], "Software License Agreement": [[11, "software-license-agreement"]], "Achieving High Performance": [[12, "achieving-high-performance"]], "General Advice": [[12, "general-advice"]], "Memory Management": [[12, "memory-management"]], "Kernel Fusion": [[12, "kernel-fusion"]], "Advanced": [[12, "advanced"]], "Further Reading": [[12, "further-reading"]], "References": [[12, "references"]], "Release Notes": [[13, "release-notes"]], "1.1.0": [[13, "id1"]], "New Features": [[13, "new-features"], [13, "id3"]], "Known Issues": [[13, "known-issues"], [13, "id5"]], "1.0.0": [[13, "id2"]], "Resolved Issues": [[13, "resolved-issues"]], "0.3.1": [[13, "id4"]], "Requirements and Functionality": [[14, "requirements-and-functionality"]], "Requirements": [[14, "requirements"]], "Supported Compilers": [[14, "supported-compilers"]], "Supported Functionality": [[14, "supported-functionality"]]}, "indexentries": {}})