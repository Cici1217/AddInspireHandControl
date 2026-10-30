"use strict";
const $ = id => document.getElementById(id);
const fmt = ConsoleI18n.format;
let state, config, plan, feature = "body", mode = "sim", revision = 0, timer, selectedProfile, view = "function";
let savedConfig, saveQueue = Promise.resolve();
const labels = {sim: "仿真", real: "真机", local: "本机"};
const statusLabels = {running:"运行中",stopping:"停止中",stopped:"已停止",completed:"已完成",failed:"失败"};
let runtimeData = {jobs:[]}, selectedJobId = null, terminalOpen = false, pollPending = false;
let restoredRuntime = false, keyboardJobId = null, keyboardSending = false;
let sessionId, sessionExpired = false;
let refreshDatasets = null;
let commonInstallation = false, commonReturnFeature = 'body', commonReturnMode = 'sim';
const sessionMessage = '网页服务已重启，请刷新页面后继续操作。';
const plans = new Map();
const bodyStages = new Map();
const startedPico = new Set();
function absolutePath(value, base=state.checkout){
  const parts=[];
  for(const part of (value.startsWith('/')?value:`${base}/${value}`).split('/')){
    if(part==='..')parts.pop();else if(part&&part!=='.')parts.push(part);
  }
  return '/'+parts.join('/');
}
function projectPath(value){
  if(!value)return value;
  const root=absolutePath(config['local.root']);
  const path=absolutePath(value,root);
  return path===root?'.':path.startsWith(root+'/')?path.slice(root.length+1):path;
}
const node = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) { if (["pre","code"].includes(tag)) e.textContent = text; else ConsoleI18n.text(e, text); } if (cls) e.className = cls; return e; };
function message(text = "") { ConsoleI18n.text($("message"), text);if(text&&terminalOpen)ConsoleI18n.text($("terminal-notice"), text); }
async function api(path, data) {
  if(sessionExpired)throw new Error(sessionMessage);
  const response = await fetch("/api/" + path, data === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json","X-Console-Token":state.token},body:JSON.stringify(data)});
  const currentSession=response.headers.get('X-Console-Session');
  if(sessionId&&currentSession&&currentSession!==sessionId){
    sessionExpired=true;clearTimeout(timer);setKeyboard(null);
    $('session-notice').hidden=false;
    ConsoleI18n.text($('connection'), '需要刷新网页');$('connection').classList.add('disconnected');
    throw new Error(sessionMessage);
  }
  if(sessionExpired)throw new Error(sessionMessage);
  if(currentSession)sessionId=currentSession;
  const result = await response.json(); if (!response.ok) throw new Error(result.error || response.statusText); return result;
}
function action(label, fn, cls) { const b = node("button",label,cls); b.onclick = async () => { b.disabled = true; try { await fn(); } catch(e) { message(e.message); } finally { b.disabled=false; } }; return b; }
function featureInfo() { return state.features.find(f => f.id === feature); }
function deviceKey(){return mode==="real"?"task.real_hand":"task.body_hand";}
function availableDevices(){
  const capability=({body:"body",hands:"control",scene:"scene",teleop:"teleop",capture:"capture",replay:"dataset_replay",vla:"inference"})[feature];
  return (state.devices||[]).filter(d=>d.deployments[mode]&&(!capability||d.deployments[mode].capabilities.includes(capability)));
}
function selectedDevice(){return availableDevices().find(d=>d.device_id===config[deviceKey()]);}
function normalizeDevice(){
  if(feature==='setup')return;
  const choices=availableDevices();
  if(choices.length&&!choices.some(d=>d.device_id===config[deviceKey()]))config[deviceKey()]=choices[0].device_id;
  const actions=selectedDevice()?.integration.hand_actions||{};
  if(config["task.hand_action"]!==undefined&&!(config["task.hand_action"] in actions))config["task.hand_action"]=Object.keys(actions)[0]||"sequence";
  if(!Object.keys(actions).length&&feature==="hands")config["task.pose"]="targets";
}
function renderNavigation() {
  $("navigation").replaceChildren(...state.features.filter(x=>!x.hidden).map((x,index) => {
    const b=action("",()=>selectFeature(x.id),view === "function" && x.id===feature?"active":"");
    ConsoleI18n.attr(b, "aria-label", x.title);b.append(node("span",String(index+1).padStart(2,"0")),node("b",x.title));return b;
  }));
  $("common-nav").classList.toggle("active",view === "common");
}
function renderView() {
  const common=view === "common", f=featureInfo();
  const installHere=common&&commonInstallation&&feature==='setup';
  const destination=$(installHere?'common-installation-content':'function-home');
  if($('function-view').parentElement!==destination)destination.append($('function-view'));
  $('common-installation').hidden=!installHere;
  $("common-view").hidden=!common;$("function-view").hidden=common&&!installHere;$("mode").hidden=common;
  ConsoleI18n.text($("title"), ConsoleI18n.text($("crumb"), common?"公共配置":f.title));
  ConsoleI18n.text($("description"), common?"":f.description);
  ConsoleI18n.text($("return-feature"), "返回"+(state.features.find(x=>x.id===commonReturnFeature)?.title||f.title));
  renderNavigation();
}
function selectFeature(id) {
  commonInstallation=false;
  feature=id;view="function";const f=featureInfo();if(!f.modes.includes(mode))mode=f.modes[0];
  normalizeDevice();
  $("mode").replaceChildren(...f.modes.map(m=>{const button=action(labels[m],()=>{mode=m;selectFeature(feature);},m===mode?"active":"");button.setAttribute("aria-pressed",String(m===mode));return button;}));
  renderView();renderSettings();schedulePreview();
}
function fieldElement(f) {
  const label=node("label",undefined,"field"+(f.type==="boolean"?" boolean":""));
  label.append(node("span",ConsoleSettings.fieldLabels[f.key]||f.label));
  let input;
  const deviceField=["task.body_hand","task.real_hand"].includes(f.key);
  if(f.choices||deviceField){
    const choices=deviceField?availableDevices().map(d=>d.device_id):f.key==="task.pose"&&!Object.keys(selectedDevice()?.integration.hand_actions||{}).length?["targets"]:f.choices;
    input=node("select");for(const v of choices){const o=node("option",deviceField?state.devices.find(d=>d.device_id===v).label:(ConsoleSettings.choiceLabels[f.key]?.[v]||v));o.value=v;input.append(o);}input.value=config[f.key];
    if(deviceField&&choices.length===1)input.disabled=true;
  }else{
    input=node("input");input.type=f.type==="boolean"?"checkbox":f.type==="number"?"number":"text";
    if(f.type==="number")input.step="any";
    if(f.type==="boolean")input.checked=config[f.key];else input.value=config[f.key];
  }
  input.dataset.key=f.key;ConsoleI18n.attr(input, "title", f.key);ConsoleI18n.attr(input, "aria-label", ConsoleSettings.fieldLabels[f.key]||f.label);
  input.disabled=input.disabled||ConsoleSettings.locked.includes(f.key);
  input.oninput=()=>{
    config[f.key]=f.type==="boolean"?input.checked:f.type==="number"?(input.value===""?null:Number(input.value)):input.value;
    ConsoleI18n.text($("config-status"), "待保存");
    if(["setup.role","task.pose","task.with_body","task.body_hand","task.real_hand","task.capture_input"].includes(f.key)){normalizeDevice();renderSettings();}
    if(f.key==="task.name")renderOutputPreview();
    const checks=$('configuration-check-results');
    if(checks?.childElementCount)checks.replaceChildren(node('p','配置已更改，请重新检查。','field-help'));
    schedulePreview();
  };
  let previousRoot=config[f.key];
  input.addEventListener('focus',()=>{previousRoot=config[f.key];});
  input.addEventListener('change',()=>{
    if(rebaseConfiguredPaths(f.key,previousRoot,config[f.key])){
      for(const element of document.querySelectorAll('[data-key]')){
        if(element.tagName==='INPUT'&&element.type==='text'&&element.dataset.key in config)element.value=config[element.dataset.key];
      }
      schedulePreview();
    }
    previousRoot=config[f.key];
  });
  if(['local.model','local.groot_root','local.tensorrt','local.xrt_root','local.scene_root'].includes(f.key)){
    const row=node('span',undefined,'path-input');
    row.append(input,action('选择目录',()=>modelDirectoryPicker(input,f),'quiet'));
    label.append(row);
  }else label.append(input);
  const hint=ConsoleSettings.hints[f.key];
  if(hint&&(f.key!=="task.headless"||selectedDevice()?.deployments[mode]?.capabilities.includes("body")))label.append(node("small",hint,"field-help"));
  return label;
}
function rebaseConfiguredPaths(key,before,after){
  const links={
    'g1.root':['g1.body_root','g1.body_script','g1.hand_root','g1.hand_script','g1.camera_root','g1.camera_python','g1.sonic_motion','setup.transfer_target','setup.backup_target'],
    'g1.body_root':['g1.body_script'], 'g1.hand_root':['g1.hand_script'],
    'g1.camera_root':['g1.camera_python','setup.transfer_target'],
    'local.groot_root':['local.policy_python'], 'local.xrt_root':['local.xrt_program'],
  };
  if(!links[key]||!before||!after||before===after)return false;
  const from=before.replace(/\/+$/,''),to=after.replace(/\/+$/,'');
  if(!from||!to)return false;
  let changed=false;
  for(const target of links[key]){
    if(config[target]===from||config[target]?.startsWith(from+'/')){
      config[target]=to+config[target].slice(from.length);changed=true;
    }
  }
  return changed;
}
function modelDirectoryPicker(input,field){
  const model=field.key==='local.model',title=model?'选择 GR00T 模型目录':'选择'+field.label;
  const dialog=node('dialog',undefined,'directory-picker');
  ConsoleI18n.attr(dialog, 'aria-label', title);
  dialog.append(node('h2',title),node('p',model?'选择本地电脑上包含 config.json 和模型权重的目录。':'选择本地电脑上的安装目录；项目内路径自动保存为相对路径。','section-note'));
  const shortcuts=node('div',undefined,'directory-shortcuts');
  const toolbar=node('div',undefined,'directory-toolbar'),path=node('input');
  path.type='text';ConsoleI18n.attr(path, 'aria-label', '浏览目录路径');
  const listing=node('div',undefined,'directory-list'),status=node('p','','field-help');
  status.setAttribute('role','status');
  const selected=node('p','','directory-selected');
  const footer=node('div',undefined,'directory-footer');
  let current=null,request=0;
  const choose=action('选择此目录',()=>{
    if(!current)return;
    input.value=projectPath(current.path);input.dispatchEvent(new Event('input',{bubbles:true}));
    input.dispatchEvent(new Event('change',{bubbles:true}));
    dialog.close();
  });
  choose.disabled=true;
  const up=node('button','上一级','quiet');
  up.onclick=()=>{if(current?.parent)navigate(current.parent);};
  up.disabled=true;
  async function navigate(destination){
    const version=++request;
    choose.disabled=true;up.disabled=true;path.disabled=true;current=null;
    ConsoleI18n.text(selected, '');listing.replaceChildren();ConsoleI18n.text(status, '正在读取目录…');
    try{
      const result=await api('directories',{path:destination,root:config['local.root']});
      if(version!==request||!dialog.isConnected)return;
      current=result;path.value=projectPath(result.path);ConsoleI18n.text(selected, fmt('当前目录：{0}',projectPath(result.path)));
      listing.replaceChildren(...result.directories.map(entry=>{
        const button=action('',()=>navigate(entry.path),'directory-entry');ConsoleI18n.raw(button,entry.name);
        ConsoleI18n.attr(button, 'aria-label', fmt('打开目录 {0}',entry.name));return button;
      }));
      ConsoleI18n.text(status, result.directories.length?'':'此目录没有可见子文件夹，可以选择当前目录。');
      choose.disabled=false;up.disabled=!result.parent;path.disabled=false;
    }catch(error){
      if(version===request&&dialog.isConnected){ConsoleI18n.text(status, error.message);path.value=destination;path.disabled=false;}
    }
  }
  for(const [title,directory] of [...(model?[['微调模型','outputs/checkpoints/finetuned'],['原始模型','outputs/checkpoints/pretrained']]:[]),['项目目录','.']]){
    shortcuts.append(action(title,()=>navigate(directory),'quiet'));
  }
  toolbar.append(up,path,action('前往',()=>navigate(path.value),'quiet'));
  path.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();navigate(path.value);}};
  footer.append(action('取消',()=>dialog.close(),'quiet'),choose);
  dialog.append(shortcuts,toolbar,listing,status,selected,footer);
  dialog.addEventListener('close',()=>{request++;dialog.remove();},{once:true});
  document.body.append(dialog);dialog.showModal();
  navigate(input.value.trim()||(model?'outputs/checkpoints':'.'));
}
function openInitialization(role){
  config['setup.role']=role;feature='setup';mode='local';view='common';commonInstallation=true;
  renderView();renderSettings();schedulePreview();
  $('common-installation').open=true;
  requestAnimationFrame(()=>$('common-installation').scrollIntoView({block:'start'}));
}
function downloadNotes(group){
  const entries=ConsoleSettings.downloads[group]||[];
  if(!entries.length)return null;
  const details=node('details',undefined,'download-notes');
  details.append(node('summary','下载／获取说明（'+entries.length+' 项）'),
    node('p','展开需要的项目，查看来源、保存位置和配置项。复制命令后在电脑终端执行；项目内命令从项目根目录开始。','field-help'));
  for(const entry of entries){
    const article=node('article');
    article.append(node('h4',entry.title),node('p',entry.text));
    if(entry.links?.length){
      const links=node('p',undefined,'download-links');
      for(const [label,url] of entry.links){
        const link=node('a',label);link.href=url;link.target='_blank';link.rel='noopener noreferrer';links.append(link);
      }
      article.append(links);
    }
    for(const command of entry.commands||[]){
      const text=command.lines.join('\n');
      article.append(node('p',command.label,'download-command-label'),node('pre',text,'download-command'),
        action('复制命令',()=>copy(text),'quiet'));
    }
    article.append(node('p',entry.location,'download-location'));details.append(article);
  }
  return details;
}
function focusCommon(group,key){
  const section=$('common-'+group);if(!section)return;
  section.open=true;section.scrollIntoView({block:'start'});
  if(key)section.querySelector(`[data-key="${CSS.escape(key)}"]`)?.focus({preventScroll:true});
}
function renderOnboarding(){
  const panel=node('section',undefined,'panel onboarding');
  panel.append(node('h2','新电脑首次配置'),node('p','在本页完成新电脑配置：查看下载说明 → 填写安装位置与 G1 连接 → 安装依赖 → 检查结果。仿真与真机共用这些设置，修改后自动保存在本机。','section-note'));
  const steps=node('div',undefined,'onboarding-steps');
  for(const [title,detail,group] of [
    ['1 本机环境','项目目录和三个 Python 环境默认使用项目内位置；已有环境可改为自己的路径。','base'],
    ['2 仿真与模型','确认 SONIC 模型、TensorRT、手配置和场景；GR00T checkpoint 在 VLA 配置中选择。','sonic'],
    ['3 PICO / VLA 依赖','自行准备 XRoboToolkit、Isaac-GR00T 及推理环境，在这里填写安装位置。','pico'],
    ['4 实验室 G1','确认 SSH、身体、手桥接和相机的真实部署路径；这些路径属于机器人。','g1'],
  ]){
    const item=node('div');item.append(action(title,()=>focusCommon(group),'quiet'),node('p',detail,'field-help'));steps.append(item);
  }
  const results=node('div');results.id='configuration-check-results';results.setAttribute('aria-live','polite');
  const buttons=node('div',undefined,'onboarding-actions');
  buttons.append(action('检查本机配置路径',async()=>{
    const snapshot=JSON.stringify(config);
    results.replaceChildren(node('p','正在检查…','field-help'));
    try{
      const report=await api('config-check',{config:JSON.parse(snapshot)});
      if(!results.isConnected)return;
      if(snapshot!==JSON.stringify(config)){results.replaceChildren(node('p','配置已更改，请重新检查。','field-help'));return;}
      results.replaceChildren(node('p',fmt('已找到 {0} 项，需补充 {1} 项。这里只检查本机文件；选择下方环境，在本页继续安装和检查 Python 依赖、模型兼容性或 G1 文件。',report.found,report.missing),'section-note'));
      const missing=report.checks.filter(row=>row.status!=='found'),found=report.checks.filter(row=>row.status==='found');
      const renderRow=row=>{
        const item=node('div',undefined,'configuration-result '+row.status);
        item.append(action(row.label,()=>focusCommon(row.group,row.key),'quiet'),node('span',row.detail),node('code',row.path));return item;
      };
      results.append(...missing.map(renderRow));
      if(found.length){const details=node('details');details.append(node('summary',`查看已找到的 ${found.length} 项`),...found.map(renderRow));results.append(details);}
    }catch(error){results.replaceChildren(node('p',error.message,'field-help'));}
  }),action('初始化数据与日志目录',async()=>{
    await api('output-directories',{config});message('项目输出目录已就绪，已有数据保留。');
  },'quiet'));
  const roles=node('div',undefined,'onboarding-actions');
  roles.append(node('span','安装与检查：'));
  for(const {role,label} of ConsoleSettings.environments){
    roles.append(action(label,()=>openInitialization(role),'quiet'));
  }
  panel.append(steps,buttons,results,roles,node('p','选择环境后，安装与检查步骤在本页展开；点击「执行此步」才会运行，结果显示在运行终端。','field-help'));$('common-start').replaceChildren(panel);
}
function renderCommon(target) {
  renderOnboarding();
  $("common-groups").replaceChildren(...[
    ["local","本地电脑","路径属于当前电脑；连接端口由电脑配置、经 SSH 转发。"],
    ["g1","G1 机器人","SSH 连接信息，以及机器人上的程序、相机和路径。"]
  ].map(([machine,title,note])=>{
    const block=node("section",undefined,"machine-config");block.id="common-machine-"+machine;
    block.append(node("h2",title),node("p",note,"machine-note"));
    const groups=node("div",undefined,"machine-groups");
    groups.append(...ConsoleSettings.groups.filter(g=>g.machine===machine).map(g=>{
    const section=node("details",undefined,"panel common-group");section.id="common-"+g.id;
    section.open=g.required||target===g.id;
    const summary=node("summary"),heading=node("div",undefined,"common-heading");
    heading.append(node("h3",g.title),node("small",g.scope));summary.append(heading);
    if(g.required)summary.append(node("span","基础必配","scope-badge required"));
    else summary.append(node("span","按需配置","scope-badge optional"));
    const fields=node("div",undefined,"common-fields");
    fields.append(...g.fields.filter(k=>state.fields.some(f=>f.key===k)).map(k=>fieldElement(state.fields.find(f=>f.key===k))));
    if(g.id==="scenes"){
      const path=state.devices?.[0]?.scene_directory?.replace(/\/[^/]+$/,"")||`${config["local.root"]}/scenes`;
      fields.append(node("p",fmt('实际目录：{0}/设备名/场景名/scene.xml',path),"field-help"));
    }
    if(g.id==="storage"){
      const paths=node('div',undefined,'field-help');paths.id='storage-paths';
      fields.append(paths,action('初始化目录',async()=>{
        await saveQueue;
        await api('output-directories',{config});
        message('目录已就绪，已有文件已保留。');
      }),action('打开数据采集',()=>selectFeature('capture'),'quiet'));
    }
    section.append(summary,node('p',g.description,'common-description'),fields);
    const downloads=downloadNotes(g.id);if(downloads)section.append(downloads);
    const environment=ConsoleSettings.environments.find(entry=>entry.groups.includes(g.id));
    if(environment){
      const actions=node('div',undefined,'common-install-actions');
      actions.append(action('查看安装与检查步骤',()=>openInitialization(environment.role),'quiet'));
      section.append(actions);
    }
    return section;
    }));
    block.append(groups);return block;
  }));
  renderOutputPreview();
}
function showCommon(target="base") {
  if(view!=='common'){commonReturnFeature=feature;commonReturnMode=mode;}
  view="common";renderCommon(target);renderView();
  requestAnimationFrame(()=>$("common-"+target)?.scrollIntoView({block:"nearest"}));
}
function renderOutputPreview(){
  const paths=plan?.output_directories;
  const common=$("storage-paths");
  if(common)common.replaceChildren(...(paths?Object.entries({sim:'仿真数据',real:'真机数据（电脑）',logs:'执行日志',calibration:'标定',replay:'回放记录',vla:'VLA 运行记录',pretrained:'原始模型',finetuned:'微调 checkpoint'}).map(([key,label])=>node('p',fmt('{0}：{1}',fmt(label),projectPath(paths[key])))):[node('p','正在解析目录…')]));
  const e=$("output-preview");
  if(e)ConsoleI18n.text(e, `outputs/datasets/${mode}/${config['task.name']}/`);
  const location=$('recording-location');
  const visible=feature==='capture';
  location.hidden=!visible;
  if(visible){
    const heading=node('div',undefined,'panel-heading');heading.append(node('h2','录制与回放位置'));
    const content=node('div',undefined,'recording-location-content');
    content.append(node('p',`本次录制来源：${labels[mode]}，数据保存在本地电脑的 ${mode} 目录。`,'section-note'));
    content.append(node('p','仿真录制：outputs/datasets/sim/','output-preview'),node('p','真机录制：outputs/datasets/real/','output-preview'));
    content.append(node('p','录制后，在「SONIC 动作回放」选择数据集、Episode 和身体回放方式，无需复制或移动。sim / real 表示录制来源；普通 PICO 身体与双手动作可在仿真或真机回放。仅用于还原仿真场景状态的数据只支持仿真查看。','section-note'));
    content.append(action('查看已录制数据',()=>selectFeature('replay'),'quiet'));
    location.replaceChildren(heading,content);
  }
}
function renderSettings() {
  refreshDatasets=null;
  renderNetworkHelp();
  const device=selectedDevice();
  const inlineSetup=view==='common'&&commonInstallation&&feature==='setup';
  const sections=ConsoleSettings.sections(feature,mode,config,device)
    .map(s=>inlineSetup?{...s,keys:s.keys.filter(k=>!['local.base_python','local.model'].includes(k))}:s)
    .filter(s=>s.keys.length);
  if(["body","hands","teleop","capture","vla"].includes(feature)&&device?.deployments[mode]?.capabilities.includes("body"))sections.push({title:mode==="real"?"真机准备姿态":"准备姿态（可选）",keys:["@ready_pose"],note:mode==="real"?"":"先到准备姿态，再开始遥操或 VLA；腿部继续由 SONIC 保持平衡。"});
  $("task-config").hidden=!sections.length;
  $("settings").replaceChildren(...sections.map(s=>{
    const section=node("section",undefined,"parameter-group");section.append(node("h3",s.title));
    if(s.note)section.append(node("p",s.note,"section-note"));
    section.append(...s.keys.map(k=>k==="@ready_pose"?readyPoseControls():k==="@targets"?targetControls(device):k==="@hand_action"?handActionControls(device):k==="@scene"?sceneControls(device):k==="@dataset"?datasetControls():fieldElement(state.fields.find(f=>f.key===k))));
    if(feature==="capture"&&s.keys.includes("task.name")){
      const p=node("p",undefined,"output-preview");p.id="output-preview";
      section.append(p,node('small','一批录制保存为一个 LeRobot 数据集，内部按 Episode 0、1、2…追加。再次启动同名批次才加 _001、_002，不覆盖已有数据。','field-help'));
    }
    return section;
  }));
  const hasBody=device?.deployments[mode]?.capabilities.includes("body");
  ConsoleI18n.text($("function-note"), feature==="capture"?(hasBody?"只读采集 · 结束录制后可继续遥操。":"独立末端仿真 · 每条结束关闭仿真并导出数据。"):feature==="hands"?(hasBody?"身体与手指分开运行。":"独立末端仿真，无需启动身体控制。") : "");
  renderOutputPreview();
  if(view==="common")renderCommon();
}
function renderNetworkHelp(){
  const help=$('network-help');
  help.hidden=!['teleop','capture'].includes(feature);
  if(help.hidden)return;
  const content=$('network-help-content'),command='ip -4 -br addr show up';
  content.replaceChildren(node('pre',command,'command'),action('复制查询命令',()=>copy(command),'quiet'));
  content.append(node('p','在电脑终端执行，取与 PICO 同一局域网的 IPv4（不带 /24），填入头显 PC Service → Enter。启动 XRoboToolkit 服务后，也可在头显自动发现弹窗选择电脑。','section-note'));
  if(mode==='real')content.append(node('p','G1 的 IP：在 G1 终端执行同一命令，填入「公共配置 → G1 连接 → SSH 主机 / IP」。','section-note'));
}

function handActionControls(device){
  const field={...state.fields.find(f=>f.key==="task.hand_action"),choices:Object.keys(device?.integration.hand_actions||{})};
  ConsoleSettings.choiceLabels["task.hand_action"]=Object.fromEntries(Object.entries(device?.integration.hand_actions||{}).map(([id,spec])=>[id,spec.label]));
  return fieldElement(field);
}
function datasetControls(){
  const block=node('div'),source=node('select'),choice=node('select'),hint=node('p','','field-help');
  ConsoleI18n.attr(source, 'aria-label', '数据采集来源');ConsoleI18n.attr(choice, 'aria-label', '已保存的数据集');
  for(const [value,title] of [['all','全部来源'],['sim','仿真录制'],['real','真机录制']]){
    const option=node('option',title);option.value=value;source.append(option);
  }
  const sourceLabel=node('label',undefined,'field'),choiceLabel=node('label',undefined,'field');
  sourceLabel.append(node('span','数据来源'),source);choiceLabel.append(node('span','数据集'),choice);
  const manual=node('details');manual.append(node('summary','手动填写其他目录'),fieldElement(state.fields.find(f=>f.key==='local.input')));
  let rows=[], pending=false, lastRefresh=0;
  const describe=()=>{
    const row=rows.find(r=>r.path===projectPath(config['local.input']));
    ConsoleI18n.text(hint, row?fmt('{0} · {1} 条（Episode 0–{2}） · {3}。读取位置：{4}',fmt(labels[row.source]||'未知来源'),row.episodes,row.episodes-1,fmt(row.playback==='simulation'?'按录制状态还原整个仿真场景，仅支持仿真查看':'身体与双手动作，可选择仿真或真机执行'),row.path):'也可手动填写其他录制目录。');
  };
  const fill=()=>{
    choice.replaceChildren();
    const placeholder=node('option','选择数据集');placeholder.value='';choice.append(placeholder);
    for(const row of rows.filter(r=>source.value==='all'||r.source===source.value)){
      const option=node('option',fmt('{0} · {1}',row.label,fmt(labels[row.source]||'未知来源')));option.value=row.path;choice.append(option);
    }
    choice.value=projectPath(config['local.input']);describe();
  };
  const refresh=async()=>{
    if(pending||!block.isConnected||Date.now()-lastRefresh<5000)return;
    pending=true;lastRefresh=Date.now();
    try{
      const next=(await api('datasets',{config})).filter(r=>r.playback!=='unsupported').map(r=>({...r,path:projectPath(r.path)}));
      if(!block.isConnected)return;
      const changed=JSON.stringify(next)!==JSON.stringify(rows);
      rows=next;
      const current=projectPath(config['local.input']);
      if(rows.length&&!rows.some(r=>r.path===current)&&(!current||current.endsWith('/datasets/sim/inspire_capture_000'))){
        config['local.input']=rows[0].path;config['task.episode']=0;
        manual.querySelector('input').value=rows[0].path;schedulePreview();
      }
      if(changed||!choice.options.length)fill();
    }finally{pending=false;}
  };
  choice.oninput=()=>{
    if(!choice.value)return;
    config['local.input']=choice.value;config['task.episode']=0;
    manual.querySelector('input').value=choice.value;
    const episode=document.querySelector('[data-key="task.episode"]');if(episode)episode.value=0;
    ConsoleI18n.text($('config-status'), '待保存');describe();schedulePreview();
  };
  source.oninput=()=>{
    fill();
    if(!choice.value&&choice.options.length>1){choice.selectedIndex=1;choice.oninput();}
  };
  manual.addEventListener('input',fill);
  block.append(sourceLabel,choiceLabel,hint,manual);
  refreshDatasets=()=>refresh().catch(e=>{ConsoleI18n.text(hint, e.message);});
  setTimeout(refreshDatasets,0);
  return block;
}
function readyPoseControls(){
  const block=node('div'),label=node('label',undefined,'field'),select=node('select');
  const base='gear_sonic/data/poses/g1/';
  const choices=[['','默认（原版 SONIC 流程）'],[base+'arms_raised.txt','双臂前抬（明显姿态）'],[base+'box_ready.txt','盒子任务准备姿态'],['custom','自定义姿态文件']];
  for(const [value,text] of choices){const option=node('option',text);option.value=value;select.append(option);}
  const current=projectPath(config['local.ready_pose']);
  select.value=choices.some(([value])=>value===current)?current:'custom';
  ConsoleI18n.attr(select, 'aria-label', '准备姿态');label.append(node('span','准备姿态'),select);block.append(label);
  const custom=fieldElement(state.fields.find(f=>f.key==='local.ready_pose'));
  custom.hidden=select.value!=='custom';block.append(custom);
  const poseNote=node('p','','section-note');
  const describe=()=>{ConsoleI18n.text(poseNote, select.value===base+'arms_raised.txt'?'双手向前抬到胸前，便于观察开始控制后的姿态变化。':select.value===base+'box_ready.txt'?'盒子任务使用的预设，双臂靠近身体，双手位置较低。':select.value==='custom'?'使用所选文件中的准备目标。':'不额外指定准备目标，沿用原版 SONIC 启动流程。');};
  describe();block.append(poseNote);
  select.oninput=()=>{
    describe();
    custom.hidden=select.value!=='custom';
    config['local.ready_pose']=select.value==='custom'?'':select.value;
    const input=custom.querySelector('input');if(input)input.value=config['local.ready_pose'];
    if(config['local.ready_pose']){config['local.vla_initial_pose']='';config['task.vla_keyframe']='';renderSettings();}
    ConsoleI18n.text($('config-status'), '待保存');schedulePreview();
  };
  block.append(node('small','初始化保留原版姿态。开始 SONIC 控制后才进入所选准备姿态：身体控制按 ]，PICO / VLA 按第一次 Enter。更换姿态需重新启动身体步骤。','field-help'));
  return block;
}
function sceneControls(device){
  const block=node("div"),label=node("label",undefined,"field"),select=node("select");
  block.id='task-scene-control';
  label.append(node("span","场景"));ConsoleI18n.attr(select, "aria-label", "场景");select.dataset.key="task.sim_scenes";
  const id=device?.device_id,current=config["task.sim_scenes"][id]||"";
  const choices=[{name:"",label:"设备默认场景"},...(device?.scenes||[])];
  if(current&&!choices.some(row=>row.name===current))choices.push({name:current,label:current+"（文件已不存在）"});
  for(const row of choices){const option=node("option",row.label||row.name);option.value=row.name;select.append(option);}
  select.value=current;
  const hint=node("p",undefined,"field-help scene-path");
  const updatePath=()=>{ConsoleI18n.text(hint, fmt('{0}：{1}',fmt(select.value?'场景文件':'自定义场景目录'),projectPath(device?.scene_directory||config['local.scene_root']+'/'+id)+'/'+(select.value?select.value+'/scene.xml':'')));};
  select.oninput=()=>{config["task.sim_scenes"][id]=select.value;updatePath();ConsoleI18n.text($("config-status"), "待保存");schedulePreview();};
  updatePath();
  const recorded=node('p','使用录制时的场景。场景随数据集一起还原，无需另选。','field-help recorded-scene');
  recorded.hidden=true;
  label.append(select);block.append(label,hint,recorded);
  return block;
}
function targetControls(device){
  const block=node("div",undefined,"effector-targets");
  if(!device)return block;
  const mounts=device.deployments[mode].mounts, legacy=device.integration.legacy_target_fields||{};
  const values=config["task.effector_targets"][device.device_id]||Object.fromEntries(mounts.map(m=>[m,legacy[m]?config[legacy[m]].split(/\s+/).map(Number):device.action_channels.map(c=>Math.min(c.upper,Math.max(c.lower,0)))]));
  for(const mount of mounts){
    const group=node("fieldset",undefined,"effector-mount");group.append(node("legend",mount));
    device.action_channels.forEach((channel,index)=>{
      const label=node("label",undefined,"field");label.append(node("span",`${channel.name} (${channel.unit})`));
      const input=node("input");input.type="number";input.min=channel.lower;input.max=channel.upper;input.step="any";
      input.value=values[mount]?.[index]??channel.lower;input.dataset.mount=mount;input.dataset.channel=channel.name;
      ConsoleI18n.attr(input, "aria-label", `${mount} ${channel.name} (${channel.unit})`);
      input.oninput=()=>{
        const next=JSON.parse(JSON.stringify(values));next[mount][index]=input.value===""?null:Number(input.value);
        config["task.effector_targets"][device.device_id]=next;values[mount][index]=next[mount][index];
        ConsoleI18n.text($("config-status"), "待保存");schedulePreview();
      };
      label.append(input,node("small",`${channel.lower}–${channel.upper} ${channel.unit}`,"field-help"));group.append(label);
    });block.append(group);
  }return block;
}
function schedulePreview() {
  clearTimeout(timer);revision++;plan=null;
  $("steps").querySelectorAll("button").forEach(b=>b.disabled=true);
  const ticket=revision;timer=setTimeout(()=>preview(ticket),250);
}
async function preview(ticket) {
  try {
    const devices=await api("devices",{config});if(ticket!==revision)return;
    if(JSON.stringify(devices)!==JSON.stringify(state.devices)){state.devices=devices;normalizeDevice();renderSettings();}
    const p=await api("plan",{config,feature,mode});if(ticket!==revision)return;plan=p;plans.set(p.id,p);message();renderSteps();renderOutputPreview();saveConfig();}
  catch(e){if(ticket===revision){message(e.message);$("steps").replaceChildren(node("p","请先修正配置，执行按钮已停用。","pending"));}}
}
async function copy(text) {await navigator.clipboard.writeText(text);message("已复制命令。");}
function renderSteps() {
  const savedScene=plan.steps.some(s=>s.simulation_replay),sceneControl=$('task-scene-control');
  const referenceControl=document.querySelector('#settings [data-key="task.replay_mode"]')?.closest('label');
  if(referenceControl)referenceControl.hidden=savedScene;
  if(sceneControl){
    sceneControl.querySelector('label').hidden=savedScene;
    sceneControl.querySelector('.scene-path').hidden=savedScene;
    sceneControl.querySelector('.recorded-scene').hidden=!savedScene;
    const hint=sceneControl.parentElement.querySelector('[data-key="task.headless"]')?.closest('label').querySelector('small');
    if(hint)hint.hidden=savedScene;
  }
  ConsoleI18n.text($("step-count"), `${plan.steps.length} 步`);
  $("steps").replaceChildren(...plan.steps.map((s,i)=>{
    const card=node("article",undefined,"step"),top=node("div",undefined,"step-top"),title=node("div");
    card.dataset.stepId=s.id;
    title.append(node("h3",`${String(i+1).padStart(2,"0")}  ${s.title}`),node("small",s.machine==="g1"?"G1 / SSH":"本机"),node("span","待启动","status step-status"));
    const planId=plan.id, buttons=node("div",undefined,"step-actions");
    const run=node("button","执行此步","start-step");
    run.onclick=async()=>{
      run.disabled=true;card.dataset.starting="true";delete card.dataset.startError;ConsoleI18n.text(run, "启动中…");updateStepStates();
      try {
        const job=await api("start",{plan_id:planId,step_id:s.id});
        setKeyboard(null);selectedJobId=job.id;setTerminalOpen(true);await poll();
      } catch(e) {
        card.dataset.startError=e.message;updateStepStates();
        card.querySelector(".step-notice").scrollIntoView({block:"nearest"});
      } finally {
        delete card.dataset.starting;ConsoleI18n.text(run, "执行此步");updateStepStates();
      }
    };
    const show=action("查看终端",()=>{const j=latestJob(planId,s.id);if(j)selectJob(j.id);});show.className="show-terminal";show.hidden=true;
    buttons.append(run,show);top.append(title,buttons);card.append(top);
    const notice=node("p",undefined,"step-notice");notice.hidden=true;notice.setAttribute("role","status");notice.setAttribute("aria-live","polite");card.append(notice);
    const instructions=operationSteps(s,mode);
    if(instructions.length){const help=node("details",undefined,"step-help"),list=node("ol",undefined,"step-guide");help.append(node("summary","操作说明"));list.append(...instructions.map(text=>node("li",text)));help.append(list);card.append(help);}
    const detail=node("details",undefined,"step-command");detail.append(node("summary","命令与说明"));
    if(s.note)detail.append(node("p",s.note,"hint"));
    detail.append(node("pre",s.launch_command,"command"));
    detail.append(node("div",fmt("工作目录：{0}\n环境：{1}\n输出：{2}",s.cwd,Object.entries(s.env).map(([k,v])=>`${k}=${v}`).join(" "),s.outputs.join("，")||plan.log_root),"step-meta"));
    if(s.nested_commands)detail.append(node("pre",s.nested_commands.join("\n"),"command"));
    detail.append(action("复制命令",()=>copy(s.launch_command)));card.append(detail);return card;
  }));
  updateStepStates();
}
function latestJob(planId,stepId){
  const step=plans.get(planId)?.steps.find(s=>s.id===stepId);
  const shared=runtimeData.jobs.filter(j=>step?.service&&j.service&&["running","stopping"].includes(j.state)&&j.step_id===stepId&&j.command===step.launch_command&&(j.control_mode||"")===(step.control_mode||""));
  return shared.at(-1)||runtimeData.jobs.filter(j=>j.plan_id===planId&&j.step_id===stepId).at(-1);
}
function updateStepStates(){
  if(!plan)return;
  for(const card of $("steps").querySelectorAll(".step")){
    const j=latestJob(plan.id,card.dataset.stepId),active=j&&["running","stopping"].includes(j.state);
    card.classList.toggle("is-running",!!active);card.classList.toggle("is-failed",j?.state==="failed");
    const badge=card.querySelector(".step-status");ConsoleI18n.text(badge, j?(j.ready?"已就绪":statusLabels[j.state]):"待启动");badge.className="status step-status "+(j?.state||"");
    card.querySelector(".start-step").disabled=!!active||card.dataset.starting==="true";card.querySelector(".show-terminal").hidden=!j||j.terminal_closed;
    const step=plan.steps.find(s=>s.id===card.dataset.stepId);
    const missing=(step.depends||[]).filter(id=>{
      const dependency=latestJob(plan.id,id);
      return !dependency||!(dependency.state==="running"&&dependency.ready||dependency.state==="completed"&&!dependency.service);
    });
    const notice=card.querySelector(".step-notice");
    if(!missing.length&&card.dataset.startError?.startsWith("前置步骤"))delete card.dataset.startError;
    ConsoleI18n.text(notice, card.dataset.startError||(!active&&missing.length?missing.map(id=>{
      const index=plan.steps.findIndex(s=>s.id===id),dependency=latestJob(plan.id,id),prefix=`第 ${String(index+1).padStart(2,"0")} 步「${plan.steps[index].title}」`;
      return dependency?.state==="running"?`${prefix}已启动，请在其终端点击「${readyLabel(id)}」。`:`请先启动${prefix}，并在其终端确认就绪。`;
    }).join(" "):""));
    notice.hidden=!notice.textContent;notice.classList.toggle("error",!!card.dataset.startError);
  }
}
function setTerminalOpen(open){
  terminalOpen=open;$("runtime").classList.toggle("collapsed",!open);document.body.classList.toggle("terminal-open",open);
  $("terminal-content").hidden=!open;$("terminal-toggle").setAttribute("aria-expanded",String(open));
  ConsoleI18n.text($("terminal-toggle-label"), open?"隐藏":"显示");
  if(!open){setKeyboard(null);$("runtime").classList.remove("maximized");ConsoleI18n.text($("terminal-expand"), "全屏");}
}
function selectJob(id){
  if(!runtimeData.jobs.some(j=>j.id===id&&!j.terminal_closed))return;
  if(selectedJobId!==id)setKeyboard(null);selectedJobId=id;setTerminalOpen(true);renderJobs(runtimeData);
  requestAnimationFrame(()=>$("job-"+id)?.querySelector(".job-output>pre")?.focus({preventScroll:true}));
}
function closeLabel(job){return job.close_requested?"等待退出…":["running","stopping"].includes(job.state)?"停止并关闭":"关闭终端";}
function closeButton(job,compact=false){
  const b=node("button",compact?"×":closeLabel(job),compact?"tab-close":"close-terminal");
  b.disabled=!!job.close_requested;ConsoleI18n.attr(b, "title", closeLabel(job)+" · "+job.title);ConsoleI18n.attr(b, "aria-label", closeLabel(job)+" · "+job.title);
  b.onclick=async()=>{
    b.disabled=true;
    try{
      const result=await api("close",{job_id:job.id});
      ConsoleI18n.text($("terminal-notice"), result.terminal_closed?"终端已关闭。":"正在停止，退出后自动关闭。");
      if(!result.terminal_closed)selectJob(job.id);
      await poll();
    }catch(e){message(e.message);}
    finally{b.disabled=!!runtimeData.jobs.find(j=>j.id===job.id)?.close_requested;}
  };
  return b;
}
function inputLabel(value,job){
  if(value==="\n"&&job?.step_id==="teleop"&&job.preparation==="ready")return "进入准备姿态（Enter）";
  if(job?.step_id==='capture'&&value!=='\n')return ({'success\n':'本条成功','failure\n':'本条失败','unlabeled\n':'暂不标注'})[value]||value;
  if(value==="\n"&&job?.step_id==="vla")return job.vla?.stage==="standing"?"开始推理（Enter）":"启动站立（Enter）";
  if(value==="\n"&&job?.step_id==="dataset_replay")return job.replay?.stage==="standing"?"开始回放（Enter）":"启动站立（Enter）";
  if(value==="\n"&&job?.step_id==="body")return sonicMode(job)==="planner"?"切到动作回放（Enter）":"切到行走 / 站立（Enter）";
  if(value==="\n")return ({teleop:"开始遥操（Enter）",capture:"开始录制（Enter）"})[job?.step_id]||"发送 Enter";
  return ({"]":"进入控制（]）", "y\n":"确认部署（y + 回车）", "O":"退出控制 O"})[value]||value.toUpperCase();
}
async function sendInput(job,value){
  await api("input",{job_id:job.id,value,...(job.step_id==='capture'?{capture_episode:job.capture?.episode}:{})});ConsoleI18n.text($("terminal-notice"), `已发送 ${inputLabel(value,job)}`);
  if(job.step_id==='capture')await poll();
}
function jobStep(job){return plans.get(job.plan_id)?.steps.find(s=>s.id===job.step_id)||job;}
function manualReady(job){const s=jobStep(job);return (job.service??s.service)&&!(job.ready_pattern??s.ready)&&!job.ready;}
function readyLabel(step){return ({scene:"确认仿真已显示",body:"确认已站稳",hand:"确认手桥接正常",tunnel:"确认转发正常",xrt:"确认头显已连接"})[step]||"确认就绪";}
function operationSteps(step,runMode){
  const id=step.step_id||step.id, sim=runMode==="sim";
  if(step.effector_only){
    if(id==="scene"||id==="send_hand")return ["点击「执行此步」，程序启动独立末端仿真并执行所选操作。关闭仿真窗口或点击「停止」结束。"];
  if(id==="teleop")return ["先连接头显并开启手部跟踪，再执行本步骤；出现 PICO ready 后手部动作开始映射到仿真。","如提示标定，按终端要求完成；停止按钮会关闭本步骤的仿真与遥操。"];
    if(id==="capture")return ["执行本步骤，看到 Capture ready 后点击「开始录制（Enter）」。","程序启动独立仿真，采集实际反馈和相机图像，结束后关闭仿真并导出数据。","每条保存后，在本终端选择「本条成功」「本条失败」或「暂不标注」，准备好后，再点「开始录制（Enter）」开始下一条。"];
    if(id==="xrt")return ["启动 PC Service，头显连接电脑并开启 Hand 和 Send；确认连接后继续。"];
  }
  if(id==="scene")return ["等电脑弹出 MuJoCo 仿真窗口，先保留支撑，不按 9。此时看起来站着，不代表身体控制已启动。","回到网页，选择本步骤的终端，点击「确认仿真已显示」。"];
  if(id==="body"){
    const rows=sim?["右侧选择「SONIC 身体控制」终端。出现 Proceed with deployment? [Y/n] 时，点击「确认部署 y↵」。按钮会发送 y 并回车，不用输入 yes。"]:["先让吊架保护机器人；右侧选择「G1 SONIC 身体控制」终端，等待 Init Done。本入口不询问部署确认。"];
    if(step.vla_control)return [...rows,"看到 Init Done 后，执行下面的「VLA 身体与双手控制」。身体终端不按 ] 或 Enter；在 VLA 终端启动站立。"];
    if(step.dataset_replay)return [...rows,"看到 Init Done 后，执行下面的「身体与双手回放」。身体终端不按 ] 或 Enter；在回放终端点击「启动站立（Enter）」。"];
    if(step.control_mode==="reference")return [...rows,
      "等 Init Done，点「进入控制 ]」，等待 CONTROL state。保持动作回放模式，不点 Enter。",
      "在右侧用「上一动作 P / 下一动作 N」选择；机器人会跟随所选动作的首帧。",
      sim?"保持仿真支撑至少 3 秒；观察姿态稳定后，点击 MuJoCo 窗口按 9 释放支撑。":"保持吊架保护，确认机器人姿态稳定后，让双脚逐步承重。",
      "点「播放 T」播放到末尾；「回到首帧 R」会重置姿态。换目录要先关闭此身体终端，再重新执行。"];
    if(step.inputs.includes("1")){
      if((step.argv||[]).includes('--ready-pose'))rows.push("看到 Init Done 后，点击「进入控制 ]」。本次会自动进入 SONIC 站立准备模式，不需要再点 Enter 切换模式。");
      else rows.push("在身体终端看到 Init Done 后，点击「进入控制 ]」。","同一终端看到 CONTROL state 后，点击「站立 / 行走 ↵」。");
      rows.push(sim?"等日志出现 motion name is planner_motion，保持支撑再等至少 3 秒。点击电脑上的 MuJoCo 仿真窗口，再按键盘 9 一次释放支撑；不要点网页的「减速 9」。":"等日志出现 motion name is planner_motion，观察机器人已经稳定站立，再按现场操作逐步让双脚承重；真机没有按 9 释放吊架的操作。");
      rows.push("确认机器人稳定站立后，回到网页身体终端点击「确认已站稳」。然后手动执行下一个步骤。","控制走动：在身体终端先点「慢走 1」，再点「前进 W」；点「停步 R」停下。");
      rows.push(step.title.includes("可选")?"做手指测试：保持身体程序运行，执行本页下一个手指动作步骤，不再按 Enter。":"做手指测试：保持身体运行，左侧选「手指控制」，取消可选身体步骤；只执行手指动作步骤。");
    }else rows.push("等 Init Done 后，继续下面的 XRoboToolkit 和 PICO 步骤。身体终端不点 ] 或 Enter；由 PICO 的「开始遥操（Enter）」启动身体。",sim?"保留仿真支撑，释放时机见下面 PICO 步骤。":"保持吊架保护，开始遥操的姿态与操作见下面 PICO 步骤。");
    return rows;
  }
  if(id==="xrt")return ["在电脑终端运行 hostname -I 或 ip -4 -br addr show up，找到与 PICO 同一局域网的电脑 IPv4（不带 /24）。","等电脑的 XRoboToolkit 服务窗口打开，再打开头显应用；在自动发现弹窗选择电脑，或在 PC Service → Enter 填写电脑 IP，看到 WORKING。","头显勾选 Head、Hand、Send，Mode 选 Full-body；回到本步骤终端，点击「确认头显已连接」。"];
  if(id==="teleop"&&(step.argv||[]).includes('--ready-pose-values'))return ["本次使用自定义准备姿态。看到 PICO_PREPARE_READY 后点「进入准备姿态（Enter）」。","等待 PICO_PREPARED，确认站稳并按现场方式释放支撑；摆好人体标定姿势后，再点「开始遥操（Enter）」。","程序检查腰和双臂到位后才切换遥操。等待期间 SONIC 保持准备姿态，腿部负责平衡。"];
  if(id==="teleop")return ["右侧选择本步骤的 PICO 终端，等待 PICO ready。已有标定文件会自动读取；若出现手势标定提示，按日志摆好双手后点「采集此姿态（Enter）」。","戴好头显，站直、双脚并拢，上臂贴身、肘弯 90°、前臂朝前、掌心相对；点击本终端的「开始遥操（Enter）」。",sim?"切到网页的身体终端，等 CONTROL state，观察 MuJoCo 中机器人已跟随。保持支撑等待至少 3 秒，再点击 MuJoCo 窗口，按键盘 9 一次释放支撑。":"查看网页身体终端的 CONTROL state，确认真机稳定站立并跟随；吊架保持保护，按现场操作让双脚承重。","轻动手臂和手指，观察机器人跟随。需要采集时，再到录制终端点「开始录制（Enter）」。"];
  if(id==="capture")return ["先在 PICO 终端开始遥操，确认机器人跟随；仿真完成释放支撑后再录制。","切到「按时长录制数据」终端，看到 Capture ready 后点击「开始录制（Enter）」。等待设定时长后自动保存。","每条保存后，在本终端选择「本条成功」「本条失败」或「暂不标注」，准备好后，再点「开始录制（Enter）」开始下一条。等待标注不停止 PICO 遥操。"];
  if(id==="vla_check")return ["先检查 checkpoint 和本机 GR00T 运行环境；检查不连接机器人。"];
  if(id==="policy")return ["等待 Server ready，再执行后续步骤。模型服务运行在电脑上。"];
  if(id==="vla")return ["在本步骤终端看到 VLA_READY 后，点击「启动站立（Enter）」。",
    sim?"等待 VLA_STANDING，确认站稳；点击 MuJoCo 窗口按 9 释放支撑。":"等待 VLA_STANDING，保留吊架保护，确认站稳后按现场操作让双脚承重。",
    "点击本终端的「开始推理（Enter）」。正常结束后身体回到站立；停止或报错会退出本轮控制，需重新启动身体步骤。"];
  if(id==="replay_check")return [step.simulation_replay?"检查保存的场景和物理状态。通过后执行下一步即可打开回放窗口。":"选择数据集并执行检查。检查完成后再启动下面的仿真或真机服务。"];
  if(id==="dataset_replay"&&step.simulation_replay)return ["执行后自动打开录制时的仿真场景，播放所选 Episode；无需启动 SONIC 或 PICO。","关闭窗口或点击「停止 Ctrl+C」结束。"];
  if(id==="dataset_replay")return ["右侧选择「身体与双手回放」终端，看到 REPLAY_READY 后点击「启动站立（Enter）」。",
    sim?"等待 REPLAY_STANDING，确认身体稳定站立。点击 MuJoCo 窗口，按 9 一次释放支撑；再观察站立是否稳定。":"等待 REPLAY_STANDING，保留吊架保护，确认站立稳定后按现场操作让双脚承重。",
    "回到回放终端，点击「开始回放（Enter）」。双手先平滑对齐首帧，再同步播放身体和手指；无需 PICO。",
    "播放完成后身体保持站立。中途点击「停止 Ctrl+C」会停止双手并退出身体控制；重新播放前需重新启动已退出的身体步骤。"];
  if(id==="hand")return ["在本步骤终端查看左右手启动日志，无串口错误后点击「确认手桥接正常」。"];
  if(id==="tunnel")return ["SSH 转发正常时可能没有输出；确认没有连接失败或端口占用错误后，点击本终端的「确认转发正常」。"];
  if(id==="camera_update")return ["先停止 G1 相机服务，确认远端目标目录已创建，再执行更新。"];
  if(id==="send_hand")return ["需要身体站立时先完成身体步骤；已经站稳就保持身体程序运行。点击本步骤的「执行此步」。","手指程序自动执行所选动作，不需要再按 Enter。只测手指可跳过可选的身体步骤，仿真保留支撑。"];
  return [];
}
function bodyStage(job){
  const text=job.output||"",events=job.startup_events||[];
  let stage=text.includes("Proceed with deployment?")||events.includes("deployment_confirm")?1:0;
  if(text.includes("Init Done")||events.includes("init_done"))stage=2;
  if(text.includes("CONTROL state")||events.includes("body_control"))stage=3;
  if(text.includes("motion name is planner_motion")||events.includes("planner_ready")||job.ready&&job.inputs.includes("1"))stage=4;
  stage=Math.max(stage,bodyStages.get(job.id)||0);bodyStages.set(job.id,stage);return stage;
}
function picoStarted(job){
  if(job.output.includes("[ThreePointPose] Calibration completed")||job.output.includes("[PoseLoop] FPS:")||job.output.includes("End-effector control active")||job.startup_events?.includes("pico_started")||job.startup_events?.includes("effector_started"))startedPico.add(job.id);
  return startedPico.has(job.id);
}
function sonicMode(job){return job.sonic?.mode||"reference";}
function terminalInputs(job){
  if(job.step_id==="teleop"&&job.preparation&&job.preparation!=="inactive")return ["ready","prepared"].includes(job.preparation)?job.inputs:[];
  if(job.step_id==='capture')return job.capture?.stage==='ready'?['\n']:job.capture?.stage==='label'?job.inputs.filter(x=>x!=='\n'):[];
  if(job.step_id==="vla")return ["ready","standing"].includes(job.vla?.stage)?job.inputs:[];
  if(job.step_id==="dataset_replay")return ["ready","standing"].includes(job.replay?.stage)?job.inputs:[];
  if(job.step_id==="teleop"&&picoStarted(job))return [];
  if(job.step_id!=="body")return job.inputs;
  const stage=bodyStage(job);
  const keys=job.inputs.includes("1")?(stage===1?["y\n"]:stage===2?["]"]:stage>=3?["\n","r","O"]:[]):stage===1?["y\n"]:stage>=2?["O"]:[];
  return keys.filter(key=>job.inputs.includes(key));
}
function picoInputLabel(job){
  const tail=job.output.slice(-1500);
  return tail.includes("press Enter to capture")&&!tail.includes("PICO ready.")?"采集此姿态（Enter）":inputLabel("\n",job);
}
const motionSets = [
  ["行走",["慢走","走路","跑步","向前跳","潜行","受伤行走"]],
  ["低姿态",["下蹲","双膝跪","单膝跪","爬行","肘部爬行"]],
  ["拳击",["拳击站姿","拳击行走","左直拳","右直拳","随机出拳","左勾拳","右勾拳"]],
  ["花式行走",["边缘行走","搬物行走","潜行 2","舞步行走","僵尸行走","持枪行走","惊吓行走"]]
];
function setKeyboard(id){
  keyboardJobId=id;
  document.querySelectorAll(".keyboard-toggle").forEach(b=>{
    const enabled=b.dataset.job===id;ConsoleI18n.text(b, enabled?"键盘已启用 · 点击停用":"启用键盘控制");
    b.setAttribute("aria-pressed",String(enabled));b.classList.toggle("primary",enabled);
  });
}
function makeBodyPad(job){
  const pad=node("div",undefined,"body-pad");pad.tabIndex=0;ConsoleI18n.attr(pad, "aria-label", "SONIC 键盘控制");
  const toolbar=node("div",undefined,"control-toolbar");
  const toggle=action("启用键盘控制",()=>{setKeyboard(keyboardJobId===job.id?null:job.id);pad.focus({preventScroll:true});},"keyboard-toggle");toggle.dataset.job=job.id;
  toolbar.append(node("strong",sonicMode(job)==="planner"?"行走控制":"动作回放"),toggle);pad.append(toolbar);
  pad.append(node("div",undefined,"native-controls"));updateBodyPad(pad,job);return pad;
}
function updateBodyPad(pad,job){
  const mode=sonicMode(job),native=job.sonic||{},groupIndex=native.motion_set||0;
  const signature=JSON.stringify([mode,groupIndex,native.motions]);
  if(pad.dataset.signature!==signature){
    pad.dataset.signature=signature;const controls=pad.querySelector(".native-controls");controls.replaceChildren();
    const keys=(items,parent=controls)=>{const row=node("div",undefined,"key-group");for(const [key,label] of items){const b=action(label,()=>sendInput(job,key),key==="r"?"movement-stop":"");b.dataset.key=key;row.append(b);}parent.append(row);};
    if(mode==="planner"){
      const group=motionSets[groupIndex]||motionSets[0];controls.append(node("h4",`${groupIndex+1}/4 · ${group[0]}`));
      keys(group[1].map((name,i)=>[String(i+1),`${name} ${i+1}`]));
      keys([["w","前进 W"],["s","后退 S"],["a","左前 A"],["d","右前 D"],[",","左移 ,"],[".","右移 ."],["q","左转 Q"],["e","右转 E"]]);
      keys([["p","上一组 P"],["n","下一组 N"],["9","减速 9"],["0","加速 0"],["-","降低 -"],["=","升高 ="]]);
      controls.append(node("p","松开方向键不会立即停止，停止移动按 R。"));
    }else{
      controls.append(node("p","读取当前动作…","current-motion"));
      keys([["p","上一动作 P"],["n","下一动作 N"],["t","播放 T"]]);
      controls.append(node("p","选动作后保持首帧；T 播放到末尾，R 重置。"));
      const list=node("details",undefined,"motion-list");list.append(node("summary",`已载入 ${(native.motions||[]).length} 个动作`));
      const names=node("ol");for(const m of native.motions||[]){const li=node("li",fmt('{0} · {1} 秒',m.name,(m.frames/50).toFixed(1)));li.dataset.motion=m.name;names.append(li);}list.append(names);controls.append(list);
    }
    const extra=node("details");extra.append(node("summary","朝向与其他按键"));
    keys(mode==="planner"?[["t","继续播放 T"],["j","参考朝向左 J"],["l","参考朝向右 L"],["i","重设朝向 I"],["z","切换编码 Z"],["f","温度 F"]]:[["q","朝向左 Q"],["e","朝向右 E"],["i","重设朝向 I"],["z","切换编码 Z"],["h","温度 H"]],extra);controls.append(extra);
  }
  const current=pad.querySelector(".current-motion");if(current)ConsoleI18n.text(current, fmt('{0} · {1}',fmt(native.playing?'播放中':'首帧 / 已结束'),native.current_motion||fmt('等待动作加载')));
  for(const item of pad.querySelectorAll("[data-motion]"))item.classList.toggle("selected-motion",item.dataset.motion===native.current_motion);
  setKeyboard(keyboardJobId);
}
document.addEventListener("keydown",async event=>{
  if(!keyboardJobId||!terminalOpen||event.ctrlKey||event.altKey||event.metaKey)return;
  if(event.target.closest("input,select,textarea,[contenteditable=true]"))return;
  if(event.key==="Escape"){setKeyboard(null);return;}
  if(event.key==="Enter"&&event.target.closest("button,summary"))return;
  const aliases={"`":"r","~":"r","<":",",">":".","_":"-","+":"=","!":"1","@":"2","#":"3","$":"4","%":"5","^":"6","&":"7","*":"8","(":"9",")":"0"};
  const value=event.key==="Enter"?"\n":event.key.toLowerCase()==="o"?"O":aliases[event.key]||event.key.toLowerCase();
  const job=runtimeData.jobs.find(j=>j.id===keyboardJobId&&j.id===selectedJobId&&j.state==="running");
  if(!job||bodyStage(job)<3)return;
  if(sonicMode(job)!=="planner"&&["`","~"].includes(event.key))return;
  const allowed=sonicMode(job)==="planner"?job.inputs:[..."pntrqeizhO","\n"];
  if(!allowed.includes(value))return;
  event.preventDefault();if(event.repeat&&!"wasd,.qe".includes(value)||keyboardSending&&!["O","r"].includes(value))return;
  keyboardSending=true;
  try{await sendInput(job,value);}catch(e){setKeyboard(null);message(e.message);}finally{keyboardSending=false;}
});
window.addEventListener("blur",()=>setKeyboard(null));
document.addEventListener("visibilitychange",()=>{if(document.hidden)setKeyboard(null);});
function renderJobs(data) {
  const previous=runtimeData;runtimeData=data;
  const visible=data.jobs.filter(j=>!j.terminal_closed);
  if(!restoredRuntime){
    restoredRuntime=true;setTerminalOpen(window.innerWidth>1050);const active=data.jobs.filter(j=>["running","stopping"].includes(j.state));
    if(active.length){selectedJobId=active.at(-1).id;setTerminalOpen(true);}
  }
  const newlyFailed=visible.find(j=>j.state==="failed"&&previous.jobs.find(old=>old.id===j.id)?.state==="running");
  if(newlyFailed){ConsoleI18n.text($("terminal-notice"), `${newlyFailed.title} 失败，请查看日志。`);setTerminalOpen(true);}
  if(!visible.some(j=>j.id===selectedJobId))selectedJobId=visible.at(-1)?.id||null;
  if(!visible.length&&previous.jobs.some(j=>!j.terminal_closed))setTerminalOpen(false);
  const active=data.jobs.filter(j=>["running","stopping"].includes(j.state)).length;
  ConsoleI18n.text($("runtime-count"), active?`${active} 个运行中`:visible.length?`${visible.length} 个终端`:"无终端");
  $("stop-all").disabled=!active;
  if(!data.jobs.some(j=>j.id===keyboardJobId&&j.id===selectedJobId&&j.state==="running"&&!j.terminal_closed))setKeyboard(null);
  const tabs=$("job-tabs"),jobs=$("jobs");
  const ids=new Set(visible.map(j=>j.id));
  for(const e of tabs.querySelectorAll(".job-tab"))if(!ids.has(e.id.slice(4)))e.parentElement.remove();
  for(const e of jobs.querySelectorAll(".job"))if(!ids.has(e.id.slice(4)))e.remove();
  if(!visible.length&&!jobs.querySelector(".empty"))jobs.append(node("p","执行一个步骤，终端会自动出现在这里。","empty"));
  for(const j of visible){
    let tab=$("tab-"+j.id);
    if(!tab){
      const wrapper=node("div",undefined,"terminal-tab");wrapper.setAttribute("role","presentation");
      tab=action("",()=>selectJob(j.id),"job-tab");tab.id="tab-"+j.id;tab.setAttribute("role","tab");tab.setAttribute("aria-controls","job-"+j.id);tab.append(node("span",undefined,"job-dot"),node("span",j.title));
      wrapper.append(tab,closeButton(j,true));tabs.append(wrapper);
    }
    tab.setAttribute("aria-selected",String(j.id===selectedJobId));ConsoleI18n.attr(tab, "title", `${j.title} · ${statusLabels[j.state]}`);tab.querySelector(".job-dot").className="job-dot "+j.state;
    const close=tab.parentElement.querySelector(".tab-close");close.disabled=!!j.close_requested;ConsoleI18n.attr(close, "title", closeLabel(j)+" · "+j.title);ConsoleI18n.attr(close, "aria-label", closeLabel(j)+" · "+j.title);
    let e=$("job-"+j.id);
    if(!e){
      jobs.querySelector(".empty")?.remove();e=node("article",undefined,"job");e.id="job-"+j.id;e.setAttribute("role","tabpanel");e.setAttribute("aria-labelledby",tab.id);
      e.append(node("div",undefined,"job-head"));
      const info=node("details",undefined,"job-info");info.append(node("summary","命令 / 日志位置"),node("div",fmt('日志：{0} · PID {1}',j.log_path,j.pid),"logpath"),node("pre",j.command,"command"));e.append(info);
      const output=node("div",undefined,"job-output"),log=node("pre","","command");log.tabIndex=0;ConsoleI18n.attr(log, "aria-label", j.title+"输出");output.append(log);e.append(output);jobs.append(e);
    }
    const wasHidden=e.hidden;e.hidden=j.id!==selectedJobId;
    // Keep controls stable while polling so clicks/focus and expanded details survive.
    const stage=j.step_id==="body"?bodyStage(j):0;
    const enterLabel=j.step_id==="teleop"?picoInputLabel(j):inputLabel("\n",j);
    const signature=JSON.stringify([j.state,j.ready,j.inputs,manualReady(j),j.close_requested,stage,sonicMode(j),enterLabel,j.replay?.stage,j.vla?.stage,j.capture?.stage,j.capture?.episode,j.preparation,j.step_id==="teleop"&&picoStarted(j)]);
    if(e.dataset.signature!==signature){
      e.dataset.signature=signature;
      const head=e.querySelector(".job-head"),controls=node("div",undefined,"job-controls"),inputs=node("div",undefined,"input-controls"),lifecycle=node("div",undefined,"lifecycle-controls");
      if(j.state==="running"){
        for(const value of terminalInputs(j)){
          const label=value==="r"?(sonicMode(j)==="planner"?"停止移动 R":"回到首帧 R"):value==="\n"?enterLabel:inputLabel(value,j);
          const shortLabel=j.step_id==="body"?({"y\n":"确认部署 y↵","]":"进入控制 ]","\n":sonicMode(j)==="planner"?"动作回放 ↵":"站立 / 行走 ↵","r":sonicMode(j)==="planner"?"停步 R":"首帧 R","O":"退出 O"})[value]:null;
          const button=action(shortLabel||label,()=>sendInput(j,value),["O","r"].includes(value)?"danger":"primary");
          ConsoleI18n.attr(button, "title", label);ConsoleI18n.attr(button, "aria-label", label);inputs.append(button);
        }
        if(manualReady(j)&&(j.step_id!=="body"||stage>=3))inputs.append(action(readyLabel(j.step_id),async()=>{await api("ready",{job_id:j.id});await poll();}));
        lifecycle.append(action("停止 Ctrl+C",async()=>{await api("stop",{job_id:j.id});await poll();},"danger"));
      }
      if(j.state==="stopping")lifecycle.append(action("强制结束",async()=>{if(window.confirm(ConsoleI18n.t("强制结束会中断未完成的数据写入，继续？"))){await api("stop",{job_id:j.id,force:true});await poll();}},"danger"));
      lifecycle.append(closeButton(j));controls.append(inputs,lifecycle);
      head.replaceChildren(node("strong",`${j.machine==="g1"?"G1":"本机"} · ${j.title}`),node("span",statusLabels[j.state],"status "+j.state),controls);
      const output=e.querySelector(".job-output");output.querySelector(".body-pad")?.remove();
      if(j.state==="running"&&j.inputs.includes("1")&&stage>=3)output.prepend(makeBodyPad(j));
    }
    const pad=e.querySelector(".body-pad");if(pad)updateBodyPad(pad,j);
    const log=e.querySelector(".job-output>pre"),nearBottom=log.scrollTop+log.clientHeight>=log.scrollHeight-40;
    if(j.step_id==='capture'){
      let panel=e.querySelector('.capture-progress');
      if(!panel){panel=node('div',undefined,'capture-progress replay-progress');e.querySelector('.job-output').prepend(panel);}
      const c=j.capture||{},names={waiting:'等待录制器就绪',ready:'准备好后按 Enter 开始本条',recording:'录制准备 / 采集与保存中',label:'已保存，等待标注任务结果',labeling:'正在保存任务结果',labeled:'任务结果已保存',complete:'本批录制完成'};
      ConsoleI18n.text(panel, `${c.episode?'第 '+c.episode+' 条 · ':''}${j.state==='running'?(names[c.stage]||c.stage):statusLabels[j.state]}${c.output?' · '+c.output:''}`);
    }
    if(j.step_id==="dataset_replay"){
      let panel=e.querySelector('.replay-progress');
      if(!panel){panel=node('div',undefined,'replay-progress');panel.append(node('span'),node('progress'));e.querySelector('.job-output').prepend(panel);}
      const r=j.replay||{},names={waiting:'等待数据检查',ready:'等待启动站立',starting:'正在启动站立',standing:'等待开始回放',aligning:'双手对齐首帧',playing:'回放中',completed:'回放完成',stopped:'已停止',failed:'回放失败'};
      const replayStage=['stopped','failed'].includes(j.state)?j.state:r.stage;
      ConsoleI18n.text(panel.querySelector('span'), `${names[replayStage]||replayStage} · ${r.frame||0}/${r.total||0} 帧 · ${r.time_s||0} 秒`);
      const bar=panel.querySelector('progress');bar.max=r.total||1;bar.value=r.frame||0;
    }
    if(j.step_id==="vla"){
      let panel=e.querySelector('.vla-progress');
      if(!panel){panel=node('div',undefined,'vla-progress replay-progress');e.querySelector('.job-output').prepend(panel);}
      const r=j.vla||{}, names={waiting:'等待初始化',ready:'等待启动站立',starting:'正在启动站立',standing:'等待开始推理',inference_starting:'等待首轮模型回复',running:'推理中',completed:'推理完成',stopped:'已停止',failed:'推理失败'};
      const stage=['stopped','failed'].includes(j.state)?j.state:r.stage;
      ConsoleI18n.text(panel, `${names[stage]||stage} · ${r.frames||0} 帧 · ${r.seconds||0} 秒`);
    }
    if(log.textContent!==j.output){if(j.output)ConsoleI18n.raw(log,j.output);else ConsoleI18n.text(log,'等待程序输出…');if(nearBottom||wasHidden)log.scrollTop=log.scrollHeight;}
  }
  updateStepStates();
}
async function poll(){
  if(pollPending||sessionExpired)return;pollPending=true;
  try{renderJobs(await api("jobs"));ConsoleI18n.text($("connection"), "网页服务正常");$("connection").classList.remove("disconnected");if(view==='function')await refreshDatasets?.();}
  catch(e){if(!sessionExpired){ConsoleI18n.text($("connection"), "网页服务断开");$("connection").classList.add("disconnected");}ConsoleI18n.text($("terminal-notice"), sessionExpired?sessionMessage:"网页服务连接中断，页面状态可能未更新。");}
  finally{pollPending=false;}
}
function setupPanelResize(){
  const handle=$("terminal-resize");let ratio=.59;
  try{ratio=Math.max(.35,Math.min(.75,Number(localStorage.getItem("sonic-terminal-width"))||.59));}catch{}
  const apply=()=>{
    const sidebar=parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sidebar"))||0;
    const available=window.innerWidth-sidebar;
    const width=Math.max(460,Math.min(available-330,available*ratio));
    document.documentElement.style.setProperty("--terminal",`${width}px`);
    handle.setAttribute("aria-valuenow",String(Math.round(ratio*100)));
  };
  const save=()=>{try{localStorage.setItem("sonic-terminal-width",String(ratio));}catch{}};
  handle.setAttribute("aria-valuemin","35");handle.setAttribute("aria-valuemax","75");
  handle.onpointerdown=event=>{
    if(event.button!==0)return;setKeyboard(null);handle.setPointerCapture(event.pointerId);document.body.classList.add("resizing");
  };
  handle.onpointermove=event=>{
    if(!handle.hasPointerCapture(event.pointerId))return;
    const sidebar=parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sidebar"))||0;
    ratio=Math.max(.35,Math.min(.75,(window.innerWidth-event.clientX)/(window.innerWidth-sidebar)));apply();
  };
  handle.onpointerup=event=>{if(handle.hasPointerCapture(event.pointerId))handle.releasePointerCapture(event.pointerId);document.body.classList.remove("resizing");save();};
  handle.onlostpointercapture=()=>document.body.classList.remove("resizing");
  handle.ondblclick=()=>{ratio=.59;apply();save();};
  handle.onkeydown=event=>{if(!["ArrowLeft","ArrowRight","Home"].includes(event.key))return;event.preventDefault();ratio=event.key==="Home"?.59:Math.max(.35,Math.min(.75,ratio+(event.key==="ArrowLeft"?.025:-.025)));apply();save();};
  window.addEventListener("resize",apply);apply();
}
function saveConfig() {
  const snapshot=JSON.stringify(config);
  saveQueue=saveQueue.then(async()=>{
    if(snapshot===savedConfig){
      if(snapshot===JSON.stringify(config))ConsoleI18n.text($("config-status"), "已自动保存");
      return;
    }
    ConsoleI18n.text($("config-status"), "正在保存…");
    await api("profiles",{name:selectedProfile,config:JSON.parse(snapshot)});
    savedConfig=snapshot;
    if(snapshot===JSON.stringify(config))ConsoleI18n.text($("config-status"), "已自动保存");
  }).catch(e=>{
    ConsoleI18n.text($("config-status"), "保存失败");
    message("配置未保存："+e.message+"。修改参数可重试。");
  });
}
async function boot(){
  setupPanelResize();
  $('reload-session').onclick=()=>window.location.reload();
  state=await api("state");if(!state.features.some(f=>f.id===feature))feature=state.features[0].id;selectedProfile=state.profiles.active;config={...state.profiles.profiles[selectedProfile]};savedConfig=JSON.stringify(config);selectFeature(feature);renderJobs(state);ConsoleI18n.text($("checkout"), fmt('当前代码目录：{0}',state.checkout));
  $("common-nav").onclick=()=>showCommon();
  $("return-feature").onclick=()=>{mode=commonReturnMode;selectFeature(commonReturnFeature);};
  $("open-terminal").onclick=()=>setTerminalOpen(true);
  $("terminal-toggle").onclick=()=>setTerminalOpen(!terminalOpen);
  $("terminal-expand").onclick=()=>{setTerminalOpen(true);const expanded=$("runtime").classList.toggle("maximized");ConsoleI18n.text($("terminal-expand"), expanded?"还原":"全屏");};
  $("log-bottom").onclick=()=>{const log=$("job-"+selectedJobId)?.querySelector(".job-output>pre");if(log)log.scrollTop=log.scrollHeight;};
  $("stop-all").onclick=async()=>{try{await api("stop-all",{});ConsoleI18n.text($("terminal-notice"), "正在按顺序停止…");await poll();}catch(e){message(e.message);}};
  $("refresh-output").onclick=async()=>{try{const data=await api("outputs");$("outputs").hidden=false;$("outputs").replaceChildren(...data.map(x=>node("p",fmt('{0}：{1}\n{2}',fmt(x.machine==='local'?'本机':'G1'),x.path,x.matches.length?x.matches.join('\n'):fmt('尚未发现输出；以任务完成日志为准。')))));}catch(e){message(e.message);}};
  await poll();setInterval(poll,1000);
}
boot().catch(e=>message("启动失败："+e.message));
