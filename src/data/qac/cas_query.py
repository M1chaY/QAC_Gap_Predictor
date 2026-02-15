"""
PubChem CAS号查询

通过SMILES查询化合物的CAS号。
"""

import urllib.parse
import requests

# 默认超时时间（秒）
DEFAULT_TIMEOUT = 30
# PubChem REST API 基础URL
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def add_halide_to_smiles(smiles: str, halide: str = 'Br') -> str:
    """
    将季铵阳离子SMILES转化为卤化盐。
    
    Args:
        smiles: QAC的SMILES字符串
        halide: 卤素类型 - 'Br', 'Cl', 'I', 'F'
    
    Returns:
        str: 添加了卤素阴离子的SMILES
    """
    halide_map = {'Br': '.[Br-]', 'Cl': '.[Cl-]', 'I': '.[I-]', 'F': '.[F-]'}
    
    if halide not in halide_map:
        raise ValueError(f"halide must be one of {list(halide_map.keys())}")
    
    return smiles + halide_map[halide]


def get_cas_number(
    smiles: str, verbose: bool = False, salt_type: str = '', timeout: int = DEFAULT_TIMEOUT
) -> str:
    """
    通过SMILES获取化合物的CAS号（使用REST API带超时）。
    
    Args:
        smiles: SMILES字符串
        verbose: 是否打印信息
        salt_type: 盐类型名称
        timeout: 超时时间（秒）
    
    Returns:
        str: CAS号字符串，不存在返回空字符串
    """
    try:
        if verbose and salt_type:
            print(f"  Querying {salt_type}...", end=' ', flush=True)
        
        # 先获取CID
        encoded_smiles = urllib.parse.quote(smiles, safe='')
        url = f"{PUBCHEM_BASE_URL}/compound/smiles/{encoded_smiles}/cids/JSON"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code != 200:
            if verbose and salt_type:
                print("Not found", flush=True)
            return ''
        
        data = response.json()
        if 'IdentifierList' not in data or not data['IdentifierList'].get('CID'):
            if verbose and salt_type:
                print("Not found", flush=True)
            return ''
        
        cid = data['IdentifierList']['CID'][0]
        
        # 获取同义词（包含CAS号）
        url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            synonyms = data.get('InformationList', {}).get('Information', [{}])
            if synonyms:
                syn_list = synonyms[0].get('Synonym', [])
                cas_numbers = [
                    s for s in syn_list 
                    if '-' in s and all(part.isdigit() for part in s.split('-'))
                ]
                if cas_numbers:
                    cas_str = ', '.join(cas_numbers[:3])  # 最多取3个CAS
                    if verbose and salt_type:
                        print(f"CAS: {cas_str}", flush=True)
                    return cas_str
        
        if verbose and salt_type:
            print("No CAS found", flush=True)
        return ''
        
    except requests.exceptions.Timeout:
        if verbose and salt_type:
            print("Timeout", flush=True)
        return ''
    except Exception as e:
        if verbose and salt_type:
            print(f"Error: {str(e)}", flush=True)
        return ''
