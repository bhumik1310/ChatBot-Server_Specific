import { Dataset,CheerioCrawler } from 'crawlee';

const crawler = new CheerioCrawler({
    async requestHandler({ request, enqueueLinks, log , body}) {
        log.info(request.url);
        await Dataset.pushData({
            url: request.url,
            
        });
        // Add all links from page to RequestQueue
        await enqueueLinks();
    },
    maxRequestsPerCrawl: 50,// Limitation for only 10 requests (do not use if you want to crawl all links)
    additionalMimeTypes:['application/pdf'] //For Image/Jpegs , add 'application/{mime_type}'
});

// Run the crawler with initial request
await crawler.run(['https://www.kamdhenuuni.edu.in/']);
await Dataset.exportToCSV('OUTPUT', { toKVS: 'my-data' }); //Outputs the links in the subdomain to the respective CSV file
