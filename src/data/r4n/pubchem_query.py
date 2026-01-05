"""
PubChem查询工具

提供PubChem数据库查询功能，使用REST API带超时控制。
"""

import urllib.parse
import requests

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

# 默认超时时间（秒）
DEFAULT_TIMEOUT = 30
# PubChem REST API 基础URL
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def validate_pubchem_compound(
    smiles: str,
    verbose: bool = False,
    idx: int = None,
    total: int = None,
    get_properties: bool = False,
    timeout: int = DEFAULT_TIMEOUT
) -> dict:
    """
    验证化合物是否在PubChem中存在（使用REST API带超时）。
    
    Args:
        smiles: SMILES字符串
        verbose: 是否打印查询进度
        idx: 当前序号
        total: 总数
        get_properties: 是否获取额外属性
        timeout: 查询超时时间（秒）
        
    Returns:
        dict: 包含CID等信息，未找到返回None
    """
    prefix = f"[{idx}/{total}] " if (idx and total) else ""
    
    try:
        if verbose:
            print(f"{prefix}Searching PubChem: {smiles}", flush=True)
        
        # 使用REST API查询CID（带超时）
        encoded_smiles = urllib.parse.quote(smiles, safe='')
        url = f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded_smiles}/cids/JSON"
        
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 404:
            if verbose:
                print(f"{prefix}Not found", flush=True)
            return None
        
        # 检查错误响应
        if response.status_code != 200:
            if verbose:
                print(f"{prefix}HTTP Error: {response.status_code}", flush=True)
            return None
        
        data = response.json()
        if 'IdentifierList' not in data or not data['IdentifierList'].get('CID'):
            if verbose:
                print(f"{prefix}Not found", flush=True)
            return None
        
        cid = data['IdentifierList']['CID'][0]
        
        # 获取基本属性
        info = {'cid': cid}
        
        if get_properties:
            info = _fetch_properties(cid, info, verbose, prefix, timeout)
        
        if verbose:
            print(f"{prefix}Found: CID={cid}", flush=True)
        return info
        
    except requests.exceptions.Timeout:
        if verbose:
            print(f"{prefix}Timeout after {timeout}s", flush=True)
        return None
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"{prefix}Request error: {e}", flush=True)
        return None
    except Exception as e:
        if verbose:
            print(f"{prefix}Error: {e}", flush=True)
        return None


def _fetch_properties(
    cid: int, info: dict, verbose: bool, prefix: str, timeout: int
) -> dict:
    """使用REST API获取额外属性（带超时）。"""
    try:
        props = "IUPACName,MolecularWeight,MolecularFormula,Complexity"
        props += ",RotatableBondCount,HeavyAtomCount,XLogP"
        url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/{props}/JSON"
        
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and data['PropertyTable'].get('Properties'):
                prop_data = data['PropertyTable']['Properties'][0]
                info.update({
                    'iupac_name': prop_data.get('IUPACName'),
                    'molecular_weight': prop_data.get('MolecularWeight'),
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


# 从 cas_query 模块导入，保持向后兼容
from src.data.r4n.cas_query import add_halide_to_smiles, get_cas_number

