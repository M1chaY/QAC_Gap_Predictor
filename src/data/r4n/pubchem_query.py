"""
PubChem查询工具

提供PubChem数据库查询功能。
"""

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False


def validate_pubchem_compound(
    smiles: str,
    verbose: bool = False,
    idx: int = None,
    total: int = None,
    get_properties: bool = False
) -> dict:
    """
    验证化合物是否在PubChem中存在。
    
    Args:
        smiles: SMILES字符串
        verbose: 是否打印查询进度
        idx: 当前序号
        total: 总数
        get_properties: 是否获取额外属性
        
    Returns:
        dict: 包含CID等信息，未找到返回None
    """
    if not PUBCHEM_AVAILABLE:
        raise ImportError("pubchempy not installed")
    
    prefix = f"[{idx}/{total}] " if (idx and total) else ""
    
    try:
        if verbose:
            print(f"{prefix}Searching PubChem: {smiles}", flush=True)
        results = pcp.get_compounds(smiles, 'smiles')
        
        if results:
            compound = results[0]
            info = {
                'cid': compound.cid,
                'iupac_name': compound.iupac_name,
                'molecular_weight': compound.molecular_weight
            }
            
            if get_properties:
                info = _add_extra_properties(info, compound.cid, verbose, prefix)
            
            if verbose:
                nm = info['iupac_name'] or '(No IUPAC Name)'
                print(f"{prefix}Found: CID={info['cid']}, Name={nm}", flush=True)
            return info
            
    except Exception as e:
        if verbose:
            print(f"{prefix}Error: {e}", flush=True)
        return None
    
    if verbose:
        print(f"{prefix}Not found", flush=True)
    return None


def _add_extra_properties(info: dict, cid: int, verbose: bool, prefix: str) -> dict:
    """添加额外的化合物属性。"""
    try:
        properties = [
            'MolecularFormula', 'Complexity', 
            'RotatableBondCount', 'HeavyAtomCount', 'XLogP',
        ]
        props = pcp.get_properties(properties, cid, namespace='cid')
        if props:
            prop_data = props[0]
            info.update({
                'molecular_formula': prop_data.get('MolecularFormula'),
                'complexity': prop_data.get('Complexity'),
                'rotatable_bond_count': prop_data.get('RotatableBondCount'),
                'heavy_atom_count': prop_data.get('HeavyAtomCount'),
                'xlogp': prop_data.get('XLogP'),
            })
    except Exception as e:
        if verbose:
            print(f"{prefix}Warning: Could not fetch properties: {e}", flush=True)
    return info


def add_halide_to_smiles(smiles: str, halide: str = 'Br') -> str:
    """
    将季铵阳离子SMILES转化为卤化盐。
    
    Args:
        smiles: R4N+的SMILES字符串
        halide: 卤素类型 - 'Br', 'Cl', 'I', 'F'
    
    Returns:
        str: 添加了卤素阴离子的SMILES
    """
    halide_map = {'Br': '.[Br-]', 'Cl': '.[Cl-]', 'I': '.[I-]', 'F': '.[F-]'}
    
    if halide not in halide_map:
        raise ValueError(f"halide must be one of {list(halide_map.keys())}")
    
    return smiles + halide_map[halide]


def get_cas_number(smiles: str, verbose: bool = False, salt_type: str = '') -> str:
    """
    通过SMILES获取化合物的CAS号。
    
    Args:
        smiles: SMILES字符串
        verbose: 是否打印信息
        salt_type: 盐类型名称
    
    Returns:
        str: CAS号字符串，不存在返回空字符串
    """
    if not PUBCHEM_AVAILABLE:
        return ''
    
    try:
        if verbose and salt_type:
            print(f"  Querying {salt_type}...", end=' ', flush=True)
        
        results = pcp.get_compounds(smiles, 'smiles')
        
        if results:
            synonyms = results[0].synonyms or []
            cas_numbers = [
                s for s in synonyms 
                if '-' in s and all(part.isdigit() for part in s.split('-'))
            ]
            
            if cas_numbers:
                cas_str = ', '.join(cas_numbers)
                if verbose and salt_type:
                    print(f"CAS: {cas_str}", flush=True)
                return cas_str
        
        if verbose and salt_type:
            print("Not found", flush=True)
        return ''
        
    except Exception as e:
        if verbose and salt_type:
            print(f"Error: {str(e)}", flush=True)
        return ''
