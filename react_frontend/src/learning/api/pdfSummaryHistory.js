const DATABASE = 'capstone-pdf-summaries';
const STORE = 'summaries';
const LIMIT = 3;

function openHistory() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withHistory(record) {
  const db = await openHistory();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE, record ? 'readwrite' : 'readonly');
      const store = transaction.objectStore(STORE);
      let recent = [];
      transaction.oncomplete = () => resolve(recent);
      transaction.onabort = () => reject(transaction.error);
      if (record) store.put(record);
      const request = store.getAll();
      request.onsuccess = () => {
        const sorted = request.result.sort((a, b) => b.createdAt - a.createdAt || b.id.localeCompare(a.id));
        recent = sorted.slice(0, LIMIT);
        if (record) {
          for (const old of sorted.slice(LIMIT)) store.delete(old.id);
        }
      };
    });
  } finally {
    db.close();
  }
}

export const readPdfSummaryHistory = () => withHistory();
export const savePdfSummary = record => withHistory(record);
