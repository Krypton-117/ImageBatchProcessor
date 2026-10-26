(() => {
  const get = id => document.getElementById(id);
  function element(tag, value, className) {
    const node = document.createElement(tag);
    node.textContent = value;
    if (className) node.className = className;
    return node;
  }
  function render(target, hits, contextIds = []) {
    target.replaceChildren();
    if (!hits.length) {
      target.append(element('p', '无匹配片段'));
      return;
    }
    for (const hit of hits) {
      const card = element('article', '', 'paper-hit');
      card.append(element('b', `${hit.title} · PDF 第 ${hit.page} 页`));
      card.append(element('p', hit.text, 'paper-excerpt'));
      const link = element('a', '打开本地原文 ↗');
      link.href = `/api/rag/papers/${encodeURIComponent(hit.source_id)}?token=${encodeURIComponent(token)}#page=${hit.page}`;
      link.target = '_blank'; link.rel = 'noopener noreferrer';
      card.append(link, element('small', ` ${hit.id}${contextIds.includes(hit.id) ? ' · 已引用' : ''}`));
      if (hit.matched_terms.length) card.append(element('small', '关联术语：' + hit.matched_terms.join('、')));
      target.append(card);
    }
  }
  window.showPaperEvidence = report => {
    const all = report.literature?.hits || [];
    const ids = report.intent?.rag_context_ids || [];
    const excerpts = new Map((report.rag_context?.excerpts || []).map(x => [x.chunk_id, x.text]));
    render(get('job-passages'), all.filter(h => ids.includes(h.id) && excerpts.has(h.id)).map(h => ({...h, text: excerpts.get(h.id)})), ids);
  };
  get('rag-form').addEventListener('submit', async event => {
    event.preventDefault();
    const query = get('rag-query').value.trim();
    if (!query) return;
    get('rag-search').disabled = true;
    get('rag-message').textContent = '正在检索本地正文…';
    try {
      const result = await api('/api/rag/search?q=' + encodeURIComponent(query));
      render(get('rag-results'), result.hits);
      get('rag-message').textContent = `${result.hits.length} 条相关片段 · 服务端检索 ${result.elapsed_ms} ms${result.cache_hit ? ' · 命中查询缓存' : ''}`;
    } catch (error) {
      get('rag-message').textContent = error.message;
    } finally { get('rag-search').disabled = false; }
  });
  api('/api/rag/status').then(s => {
    get('rag-status').textContent = s.documents ? `${s.documents} 篇论文 · ${s.pages} 页 · ${s.chunks} 个片段` : '尚无论文索引';
  }).catch(error => { get('rag-status').textContent = error.message; });
})();

document.getElementById('paper-file').addEventListener('change',async event=>{
 const file=event.target.files[0];if(!file)return;
 const status=document.getElementById('rag-status');status.textContent='索引中…';
 try{const r=await fetch('/api/rag/import',{method:'POST',headers:{'X-IBP-Token':token,'X-Filename':encodeURIComponent(file.name)},body:file});const result=await r.json();if(!r.ok)throw Error(result.error);
 const check=async()=>{try{const s=await api('/api/rag/status');if(s.import?.status==='running'){setTimeout(check,1000);return;}status.textContent=s.import?.error||s.documents+' 篇 · '+s.chunks+' 片段';}catch(e){status.textContent=e.message;}};check();
 }catch(e){status.textContent=e.message;}finally{event.target.value='';}
});
