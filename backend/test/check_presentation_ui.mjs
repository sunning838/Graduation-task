// Browser regressions with offline fixtures. --live-cache also checks one already generated image.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire(process.env.PRESENTATION_TEST_NODE_MODULES
 ? pathToFileURL(path.join(process.env.PRESENTATION_TEST_NODE_MODULES,'_entry.cjs'))
 : new URL('../../react_frontend/package.json',import.meta.url));
const {chromium}=require('playwright');
const {lectureFromSegments, MAX_SLIDES}=await import(new URL('../../react_frontend/src/learning/components/lectureSlides.js',import.meta.url));
const longSegments=Array.from({length:18},(_,index)=>({index,markdown:`### 개념 ${index}\n\n원문 설명 ${index}입니다.`}));
const longLecture=lectureFromSegments(longSegments);
assert.equal(MAX_SLIDES,16);
assert.equal(longLecture.slides.length,16);
assert.deepEqual(longLecture.slides.flatMap(slide=>slide.segment_indices),longSegments.map(segment=>segment.index));
assert.ok(longLecture.slides.every(slide=>slide.image_status==='pending'));
const base=process.env.PRESENTATION_TEST_URL || 'http://127.0.0.1:5188';
const backend=process.env.PRESENTATION_TEST_API_URL || 'http://127.0.0.1:8018';
const output=new URL('./.presentation-ui/',import.meta.url);
await fs.mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
const liveMode=process.argv.includes('--live-cache');
const live=liveMode ? JSON.parse(await fs.readFile(new URL('./.presentation-live.json',import.meta.url),'utf8')) : null;
const text='### 스택의 구조\n\n스택은 마지막에 들어온 데이터가 먼저 나오는 자료 구조입니다.\n\n- 한쪽 끝에서 데이터를 넣고 꺼냅니다.\n\n### PUSH 과정\n\nPUSH는 스택의 맨 위에 데이터를 추가하는 연산입니다.\n\n- 마지막으로 넣은 데이터가 맨 위에 있습니다.\n\n### POP 과정\n\nPOP은 스택의 맨 위에서 데이터를 꺼내는 연산입니다.\n\n- 가장 나중에 넣은 데이터를 먼저 꺼냅니다.';
const segments=[
 '### 스택의 구조\n\n스택은 마지막에 들어온 데이터가 먼저 나오는 자료 구조입니다.', '- 한쪽 끝에서 데이터를 넣고 꺼냅니다.',
 '### PUSH 과정\n\nPUSH는 스택의 맨 위에 데이터를 추가하는 연산입니다.', '- 마지막으로 넣은 데이터가 맨 위에 있습니다.',
 '### POP 과정\n\nPOP은 스택의 맨 위에서 데이터를 꺼내는 연산입니다.', '- 가장 나중에 넣은 데이터를 먼저 꺼냅니다.'
].map((markdown,index)=>({index,markdown,start:index*5,end:(index+1)*5}));
const slides=['스택의 구조','PUSH 과정','POP 과정'].map((title,index)=>({id:`slide-${index}`,title,segment_indices:[index*2,index*2+1],
 bullets:[['마지막에 들어온 데이터가 먼저 나옵니다.','입출력 위치는 스택의 맨 위입니다.'],['맨 위에 데이터를 추가합니다.'],['맨 위의 데이터를 꺼냅니다.']][index],image_status:'pending',image_url:null,caption:''}));
const id='a'.repeat(64);
let phase='idle', pollCount=0, tutorRequests=0, retries=0;
const starts=[];
const errors=[];
const page=await browser.newPage({viewport:{width:1600,height:1000}});
page.on('pageerror',error=>errors.push(error.message));
const json=(route,value,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(value)});
const variantText=(markdown,variant)=>variant==='simple' ? markdown.replaceAll('자료 구조','데이터를 담는 구조') : markdown;
function presentation(status, variant) {
 const ready=status==='ready' ? slides.length : ['generating','partial'].includes(status) ? 1 : 0;
 const failed=status==='partial' ? slides.length-ready : 0;
 return {package_id:id,status,segments:segments.map(segment=>({...segment,markdown:variantText(segment.markdown,variant)})),
  images:{total:slides.length,ready,failed},retryable:failed>0,
  slides:slides.map((slide,index)=>({...slide,image_status:index<ready ? 'ready' : failed ? 'failed' : 'pending',
   image_url:index<ready ? `/api/learning/presentations/${id}/images/${slide.id}` : null}))};
}
await page.route(url=>url.pathname.startsWith('/api/'),async route=>{
 const url=new URL(route.request().url());const path=url.pathname;
 const body=route.request().method()==='POST' ? route.request().postDataJSON() : null;
 if(path==='/api/certifications') return json(route,{certifications:[{id:'EIP',label:'정보처리기사',topics:[],option_count:4}]});
 if(path==='/api/learning/catalog') return json(route,{config:{EIP:{label:'정보처리기사',topics:{software_design:'소프트웨어 설계'}}},lessons:[
  {id:live?.lesson_id || 'test-lesson',version:'v1',cert:'EIP',subject:'software_design',title:live ? '소프트웨어 생명 주기' : '자료 구조의 스택'}]});
 if(path==='/api/learning/progress') return json(route,{});
 if(path==='/api/learning/teach') {
  if(body.question){tutorRequests++;return json(route,{text:'후입선출은 나중에 넣은 데이터를 먼저 꺼내는 방식입니다.',visual:null});}
  return json(route,{text:live?.text || variantText(text,body.variant),visual:null});
 }
 if(path.startsWith('/api/learning/presentations') && live) {
  assert.notEqual(path,'/api/learning/presentations','Cached-image checks must never send a generation request');
  assert.ok(!path.endsWith('/retry'),'Cached-image checks must never retry image generation');
  const response=await route.fetch({url:backend+path});
  return route.fulfill({response});
 }
 if(path==='/api/learning/presentations/lookup') return json(route,presentation(body.variant==='basic' ? phase : 'idle',body.variant));
 if(path==='/api/learning/presentations') {
  starts.push(body); phase='planning'; return json(route,presentation(phase));
 }
 if(path.endsWith('/retry')) { retries++; phase='ready'; return json(route,presentation(phase)); }
 if(path===`/api/learning/presentations/${id}`){
  pollCount++;
  if(phase==='planning') phase='generating';
  return json(route,presentation(phase));
 }
 if(path.startsWith(`/api/learning/presentations/${id}/images/`)) return route.fulfill({contentType:'image/svg+xml',
  body:'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90"><rect width="160" height="90" fill="lightblue"/></svg>'});
 if(path==='/api/learning/tts') return json(route,{audio_url:'/api/learning/tts/offline/audio',timeline:segments});
 if(path==='/api/learning/tts/offline/audio') {
  const rate=8000, samples=30*rate;const wav=Buffer.alloc(44+samples*2);
  wav.write('RIFF',0);wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);
  wav.writeUInt32LE(rate,24);wav.writeUInt32LE(rate*2,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(samples*2,40);
  const range=route.request().headers().range;
  if(range){
   const match=range.match(/bytes=(\d+)-(\d*)/);const start=Number(match?.[1]||0);const end=match?.[2]?Math.min(Number(match[2]),wav.length-1):wav.length-1;
   return route.fulfill({status:206,contentType:'audio/wav',headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${wav.length}`},body:wav.subarray(start,end+1)});
  }
  return route.fulfill({contentType:'audio/wav',headers:{'Accept-Ranges':'bytes'},body:wav});
 }
 return json(route,{});
});

try{
 await page.goto(base);await page.getByRole('button',{name:'학습 시작하기',exact:true}).click();
 await page.getByRole('button',{name:'처음부터 공부하기',exact:true}).click();
 await page.getByRole('region',{name:'전체 강의 본문',exact:true}).waitFor();
 assert.equal(await page.locator('.lecture-reference').isVisible(),true,'Transcript must be open initially');
 assert.equal(await page.getByRole('dialog',{name:'튜터에게 질문',exact:true}).isVisible(),false);
 if(live){
  assert.equal(live.result.status,'ready');
  assert.equal(live.result.images.total,live.result.slides.length);
  assert.equal(live.result.images.ready,live.result.slides.length);
  const imageSources=new Set();
  for(let index=0;index<live.result.slides.length;index++){
   await page.locator('.slide-dots button').nth(index).click();
   await page.locator('.slide-image-button img').waitFor();
   await page.locator('.slide-image-button img').evaluate(img=>img.decode());
   imageSources.add(await page.locator('.slide-image-button img').getAttribute('src'));
   await page.screenshot({path:new URL(`live-slide-${index+1}.png`,output).pathname.replace(/^\//,'')});
  }
  assert.equal(imageSources.size,live.result.slides.length);
  await page.locator('.slide-dots button').first().click();
 }else{
  await page.getByRole('button',{name:'설명 그림 만들기',exact:true}).waitFor();
  await page.getByText('슬라이드 3장 · 그림 3장',{exact:true}).waitFor();
  assert.equal(starts.length,0);
  assert.equal(await page.locator('.slide-image-note').count(),0);
  await page.getByRole('button',{name:'다음 →',exact:true}).click();
  assert.equal(await page.locator('.slide-content h3').textContent(),'PUSH 과정');
  await page.getByRole('button',{name:'← 이전',exact:true}).click();
  await page.getByRole('button',{name:'▶ 음성으로 듣기',exact:true}).click();
  await page.locator('audio').evaluate(audio=>new Promise(resolve=>{if(audio.readyState>=1)resolve();else audio.addEventListener('loadedmetadata',resolve,{once:true});}));
  await page.locator('audio').evaluate(audio=>{audio.preload='auto';});
  await page.waitForFunction(()=>document.querySelector('audio')?.readyState>=3);
  await page.locator('audio').evaluate(audio=>{audio.pause();audio.currentTime=21;audio.dispatchEvent(new Event('timeupdate'));});
  await page.waitForFunction(()=>document.querySelector('.slide-content h3')?.textContent==='POP 과정');
  assert.match(await page.locator('.reference-paragraph.is-speaking').textContent(),/POP/);
  await page.getByLabel('강의 재생 속도',{exact:true}).selectOption('1.5');
  assert.equal(await page.locator('audio').evaluate(audio=>audio.playbackRate),1.5);
  await page.locator('.reference-scroll').hover();await page.mouse.wheel(0,-300);
  await page.waitForFunction(()=>!document.querySelector('.lecture-follow input').checked);
  await page.getByRole('button',{name:'현재 강의 위치로',exact:true}).click();
  assert.equal(await page.locator('.lecture-follow input').isChecked(),true);
 }
 await page.getByRole('button',{name:'튜터에게 질문',exact:true}).click();
 await page.getByRole('dialog',{name:'튜터에게 질문',exact:true}).waitFor();
 await page.getByPlaceholder('이 부분이 궁금해요…').fill('후입선출이 무슨 뜻인가요?');
 await page.getByRole('button',{name:'질문하기',exact:true}).click();
 await page.getByText('후입선출은 나중에 넣은 데이터를 먼저 꺼내는 방식입니다.',{exact:true}).waitFor();
 assert.equal(tutorRequests,1);
 await page.getByRole('button',{name:'닫기',exact:true}).click();
 await page.locator('.presentation-heading').scrollIntoViewIfNeeded();
 await page.screenshot({path:new URL(live?'live-desktop.png':'desktop.png',output).pathname.replace(/^\//,'')});
 if(!live){
  assert.equal(starts.length,0,'Opening the lecture, navigating, audio and questions must not create images');
  assert.equal(pollCount,0);
  await page.getByRole('tab',{name:'쉬운 설명',exact:true}).click();
  await page.getByRole('region',{name:'전체 강의 본문',exact:true}).getByText('스택은 마지막에 들어온 데이터가 먼저 나오는 데이터를 담는 구조입니다.',{exact:true}).waitFor();
  await page.getByRole('button',{name:'설명 그림 만들기',exact:true}).waitFor();
  assert.equal(starts.length,0,'Switching explanations must not create images');
  await page.getByRole('tab',{name:'기본 설명',exact:true}).click();
  await page.getByRole('button',{name:'다음 →',exact:true}).click();
  await page.getByRole('button',{name:'설명 그림 만들기',exact:true}).click();
  await page.getByText(/1 \/ 3장 준비 완료/).waitFor();
  assert.deepEqual(starts,[{lesson_id:'test-lesson',variant:'basic',text}]);
  await page.locator('.slide-dots button').first().click();
  await page.locator('.slide-image-button img').evaluate(img=>img.decode());
  phase='partial';
  await page.getByRole('button',{name:'그림 다시 준비하기',exact:true}).waitFor({timeout:10000});
  assert.equal(await page.locator('.reference-scroll').isVisible(),true);
  await page.getByRole('button',{name:'그림 다시 준비하기',exact:true}).click();
  await page.getByText(/3 \/ 3장 준비 완료/).waitFor();
  assert.equal(retries,1);
  await page.getByRole('tab',{name:'쉬운 설명',exact:true}).click();
  await page.getByRole('button',{name:'설명 그림 만들기',exact:true}).waitFor();
  await page.getByRole('tab',{name:'기본 설명',exact:true}).click();
  await page.getByText(/3 \/ 3장 준비 완료/).waitFor();
  assert.equal(await page.getByRole('button',{name:'설명 그림 만들기',exact:true}).count(),0);
  assert.equal(starts.length,1,'Reopening a cached lecture must not regenerate its images');
 }
 await page.setViewportSize({width:1366,height:768});await page.locator('.presentation-heading').scrollIntoViewIfNeeded();
 await page.screenshot({path:new URL(live?'live-laptop.png':'laptop.png',output).pathname.replace(/^\//,'')});
 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.locator('.lecture-reference').isVisible());
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2);
 assert.equal(overflow,false,'Mobile layout must not overflow horizontally');
 await page.screenshot({path:new URL(live?'live-mobile.png':'mobile.png',output).pathname.replace(/^\//,'')});
 assert.deepEqual(errors,[]);
 console.log(JSON.stringify({checks:'manual generation, cached images, slide navigation, audio seek/speed/highlight, free reading, question drawer, retry, variants, responsive layout',liveImage:liveMode,polls:pollCount,screenshots:output.href}));
}catch(error){
 await page.screenshot({path:new URL('failure.png',output).pathname.replace(/^\//,'')});
 console.error('Browser errors:',errors,'Page:',(await page.locator('body').innerText()).slice(0,1000));
 throw error;
}finally{await browser.close();}
